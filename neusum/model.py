import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Score(nn.Module):
    def __init__(self,hq,hd,hs):
        super(Score,self).__init__()
        self.hq = hq
        self.hd = hd
        self.hs = hs
        self.linear_q = nn.Linear(hq, hs, bias=True) #h
        self.linear_d = nn.Linear(hd, hs, bias=True) #s
        self.linear_s = nn.Linear(hs, 1, bias=True) #score
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch_size,hidden_size(256) #batch_size,hidden_size
        context: batch_size,sent_len,hidden_size(256)
        t_judge：batch_size
        """
        if precompute is None:
            precompute00 = self.linear_d(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = F.tanh(tmp10)
        energy = self.linear_s(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL

        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1e8)

        return energy, precompute

class Encoder(nn.Module):
    def __init__(self,vocb_size,emd_size,hidden_size):
        super(Encoder,self).__init__()
        self.vocb_size = vocb_size
        self.emd_size = emd_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocb_size,emd_size) 
        self.gru = nn.GRU(emd_size,hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self,input,seq_length): 
        '''
        input:batch_size,sent_len,seq_len #这样的话需要事先进行填充 输入的是句子
        seq_length：batch_size,sent_len #记录每个文档中每个句子的的长度，最小为2（sos eos）
        '''
        batch_size = input.shape[0]
        sent_len = input.shape[1]
        seq_len = input.shape[2]

        #对input值进行处理，处理成维度batch_size*sent_len,seq_len
        input_change = input.view(batch_size*sent_len,seq_len)
        #对长度进行处理，处理成 batch_size*sent_len的list
        length = seq_length.view(-1).tolist()

        word_vector = self.embedding(input) # batch_size*sent_len,seq_len,emd_size
        #进行pack处理
        word_vector = word_vector.view(batch_size*sent_len,seq_len,self.emd_size)
        pack_input = pack(word_vector,length,batch_first=True,enforce_sorted=False)
        _,hidden = self.gru(pack_input) #hidden层的大小为 2,batch_size*sent_len,hidden_size
        #unpack处理
        # unpack_hidden = unpack(hidden,batch_first=True) #(batch_size*sent_len,2,hidden_size),hidden是个tensor？？
        #把hidden处理成一个文档为一行的结果 #大小为(batch_size,sent_len,2*hidden_size)
        # hidden_change = unpack_hidden[0].reshape(batch_size,sent_len,2*hidden_size)
        hidden_change = hidden.transpose(0,1).reshape(batch_size,sent_len,2*self.hidden_size)

        return hidden_change 

class DocumentEncoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(DocumentEncoder,self).__init__()
        self.input_size = input_size #这里的输入就是上一个隐藏层 256
        self.hidden_size = hidden_size #128

        self.gru = nn.GRU(input_size,hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self,input,sent_length): 
        '''
        input:batch_size,sent_len,2*hidden_size(256) #输入的是文档，每个文档里面的句子数目是不一样的
        sent_length：batch_size #记录每个句子的
        '''
        sent_len = input.shape[1]
        batch_size = input.shape[0]

        #处理长度
        length = sent_length.tolist() #-------------------
        pack_input = pack(input,length,batch_first=True,enforce_sorted=False)

        output,hidden = self.gru(pack_input) 
        unpack_output = unpack(output,batch_first=True)#(batch_size,sent_len,,2*hidden_size)
        # unpack_hidden = unpack(hidden,batch_first=True)# batch_size,2,hidden_size,backward =1
        unpack_hidden = hidden.transpose(0,1)# batch_size,2,hidden_size,backward =1

        unpack_output = self.dropout(unpack_output[0])
        unpack_hidden = self.dropout(unpack_hidden)
        return unpack_output,unpack_hidden  #产生的这个里，是pad产生的句子的结果就是0


class Select(nn.Module): 
    def __init__(self,input_szie,hidden_size,max_step,hq,hd,hs):
        super(Select,self).__init__()
        self.input_szie = input_szie
        self.hidden_size = hidden_size
        self.max_step = max_step #最大步骤
        self.hq = hq
        self.hd = hd
        self.hs = hs

        self.gru = nn.GRUCell(input_szie,hidden_size) #hidden_size = 256
        self.scorer =Score(hq,hd,hs)#...

    def gen_mask_with_length(self, doc_len, batch_size, lengths):
        mask = torch.ByteTensor(batch_size, doc_len).zero_().to(device)
        ll = lengths.data.view(-1).tolist() #看lengths处理出来是什么格式的？
        for i in range(batch_size):
            for j in range(doc_len):
                if j >= ll[i]:
                    mask[i][j] = 1
        mask = mask.float()
        return mask

    def get_hard_attention_index(self,seq_len, batch_size, indices):  #找到对应的句子的表示
        if isinstance(indices, Variable):
            index_data = indices.data
        else:
            index_data = indices
        buf = []
        for batch_id, seq_idx in enumerate(index_data):
            if seq_idx == -1: #如果是-1，那么就手动变成输入0号句子的表示
                seq_idx = 0
            idx = batch_id * seq_len + seq_idx
            buf.append(idx)
        return torch.LongTensor(buf)

    def forward(self,s0,h0,doc_encoding,tgt,sent_length): 
        '''
        doc_encoding:(batch_size,sent_len,2*hidden_size)256
        h0:batch_size,hidden_size_de(256)
        s0:batch_size,2*hidden_size(256)
        tgt:batch_size个list内容-batch_size step 里面就放置元组就行：类似于这样[[1, 2], [2, 3, 4], [1]]->补全的
        sent_length：batch_size #记录每个文档的句子数目
        '''
        sent_len = doc_encoding.shape[1]
        batch_size = doc_encoding.shape[0]
        cur_input = s0
        hiddedn = h0
        mask = self.gen_mask_with_length(sent_len, batch_size, sent_length)
        mask = Variable(mask, requires_grad=False, volatile=(not self.training)) #冻结
        self.scorer.applyMask(mask)
        All_score = []
        if self.training: #如果训练的话才会用到这个
            tgt_change = tgt.transpose(0,1)
        doc_encoding_change = doc_encoding.reshape(batch_size*sent_len,self.hidden_size)#转换后的维度
        precompute = None
        test_mask = []
        for i in range(self.max_step):
            input_vector = cur_input
            hiddedn = self.gru(input_vector,hiddedn)  #bach_size,hidden_size(256)
            score, precompute = self.scorer(hiddedn,doc_encoding,precompute)
            All_score.append(score) #step batch_size sent_len
            if self.training:
                #找到下一次要进行的输入的值，根据tgt的标号来找
                tgt_idx =  tgt_change[i] #tgt_idx = [9,2,3,-1,-1，-1,4](batch_size)
                idx = self.get_hard_attention_index(sent_len, batch_size, tgt_idx).to(device)
                idx = Variable(idx, requires_grad=False, volatile=(not self.training)) #冻结
                cur_input = doc_encoding_change.index_select(dim=0, index=idx) #找到对应的输入的contxt
            elif not self.training:
                for ii in test_mask:#-----------model new 这里进行这样的操作之后，后面就不用再进行mask了，这里的就已经被mask了
                    score[0][ii] = -1e8#-------
                max_score, max_idx = score.max(dim=1) 
                test_mask.append(max_idx) #将这个添加进去#------
                idx = self.get_hard_attention_index(sent_len, batch_size, max_idx).to(device)
                idx = Variable(idx, requires_grad=False, volatile=(not self.training)) #冻结
                cur_input = doc_encoding_change.index_select(dim=0, index=idx) #找到对应的输入的contxt
        All_score = torch.stack(All_score)
        return All_score,mask
        #loss的计算
        #S是选取的句子的内容，需要使用这个和对应的tgt计算Q
        #保存的是最大的那个分数，之后会使用这个分数计算lP
        #然后计算损失函数J即可

class DecInit(nn.Module):
    def __init__(self,hidden_size_en,hidden_size_de):
        super(DecInit,self).__init__()
        self.initer = nn.Linear(hidden_size_en,hidden_size_de)

    def forward(self,s1_backward):
        '''
        s1_backward:batch_size,1,hidden_size_en(128)
        '''
        return F.tanh(self.initer(s1_backward)) #batch_size,1,hidden_size_de(256)

class NeuSum(nn.Module):
    def __init__(self,sent_encoder, doc_encoder, decinit, select):
        super(NeuSum,self).__init__()
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder
        self.select = select
        self.decinit = decinit
    
    def forward(self,input,seq_length,sent_length,tgt=None): 
        '''
        input: batch_size,sent_len,seq_len #
        seq_length：batch_size,sent_len #记录每个文档中每个句子的的长度，最小为2（sos eos）
        sent_length：batch_size #记录每个句子的
        tgt:batch_size,5个list内容 里面就放置元组就行：类似于这样[[1, 2], [2, 3, 4], [1]]->补全？！ 对应的分数也是需要补全的

        sent_encoding:(batch_size,sent_len,2*hidden_size)
        doc_encoding:(batch_size,sent_len,2*hidden_size)
        init_hidden:(batch_size,2,hidden_size)
        h0:batch_size,hidden_size_de(256)
        s0:batch_size,2*hidden_size(256)
        '''
        sent_encoding = self.sent_encoder(input,seq_length)
        doc_encoding,init_hidden= self.doc_encoder(sent_encoding,sent_length)
        h0 = self.decinit(init_hidden[:,1,:]) #batch_size,hidden_size_de(256)
        s0 = torch.zeros(doc_encoding.shape[0],doc_encoding.shape[2]).to(device) #batch_size,2*hidden_size(256)
        s0 = Variable(s0, requires_grad=False, volatile=(not self.training)) 
        All_score,mask = self.select(s0,h0,doc_encoding,tgt,sent_length)
        
        return All_score,mask #返回的结果就是每个文档选取的句子标号和这个文档的分数