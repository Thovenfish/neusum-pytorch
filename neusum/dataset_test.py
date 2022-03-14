from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.autograd import Variable
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
def pad(data, pad_idx=-1, max_len=None):
    if max_len is None:
        max_len = max([len(instance) for instance in data])
    return [instance + [pad_idx] * max(max_len - len(instance), 0) for instance in data]

def find_martrix_max_value(data_matrix): 
        new_data=[]    
        for i in range(len(data_matrix)):        
            new_data.append(max(data_matrix[i]))
        return max(new_data)

def pad_srcs(srcs,seq_length,sent_length):
    srcs_new = copy.deepcopy(srcs) #否则就会是把原来的句子都改掉了
    sentence = [0 for i in range(seq_length)]
    sentence[0] = 2
    sentence[1] = 3
    for src in srcs_new:
        for sent in src:
            for i in range(seq_length-len(sent)):
                sent.append(0) #给每个句子增加0
        for j in range(sent_length-len(src)):
            src.append(sentence)
    return srcs_new

def pad_score(scores,sent_length):
    scores_new = copy.deepcopy(scores)
    for score in scores_new:
        for step in score:
            for i in range(sent_length-len(step)):
                step.append(0) #给每个句子增加0
    return scores_new

def pad_seq(seq_lengths,sent_length):
    seq_lengths_new = copy.deepcopy(seq_lengths)
    for seq_length in seq_lengths_new:
        for i in range(sent_length-len(seq_length)):
            seq_length.append(2) 
    return seq_lengths_new

class MyDatasetTest(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_file_doc):
        super(MyDatasetTest, self).__init__()
        self.examples = self.load_examples(src_file, tgt_file, src_vocab,tgt_file_doc)

    def load_examples(self, src_file, tgt_file, src_vocab,tgt_file_doc):
        """ 
        TODO define how to load examples from file
        '''
        src: batch_size,sent_len,seq_len #
        seq_length：batch_size,sent_len #记录每个文档中每个句子的的长度，最小为2（sos eos）
        sent_length：batch_size #记录每个句子的
        tgt:batch_size,5个list内容 里面就放置元组就行：类似于这样[[1, 2], [2, 3, 4], [1]]->补全？！ 对应的分数也是需要补全的
        score: batch*step, doc_len
        """
        print("In load examples")
        examples = []
        with open(src_file,encoding='utf-8') as fsrc, open(tgt_file,encoding='utf-8') as ftgt,open(tgt_file_doc,encoding='utf-8') as fdoc: #同时打开这两个文件
            for src_line, tgt_line, tgt_doc_line in tqdm(zip(fsrc, ftgt,fdoc)):
                if tgt_line.split()[0]=='None':
                    continue #如果不行就跳过这一行
                src = []
                seq_length = []
                sent_length = 0
                tgt_doc1 = []
                tgt_doc2 = []
                src_doc = []
                sentence = src_line.split('##SENT##') #s对文档里面的每一句话划分
                sent_length = len(sentence)
                for sent in sentence:
                    src_sent = (
                        [src_vocab.word2id("<sos>")] 
                        + [src_vocab.word2id(w) for w in sent.rstrip().split(" ")] 
                        + [src_vocab.word2id("<eos>")]
                    )
                    seq_length.append(len(sent.rstrip().split(" "))+2) #得到每句话的长度+2
                    src.append(src_sent)

                tgt_list = tgt_line.split('\t')
                tgt=[int(idx) for idx in tgt_list[0].split()] #得到想要获取的列表
                tgt_r1 = [sentence[s_idx] for s_idx in tgt] #得到结果句子列表,我们自己后期产生的那个
                tgt_doc1.append(' '.join(tgt_r1))

                tgt_r2 = tgt_doc_line.split('##SENT##')
                tgt_doc2.append(' '.join(tgt_r2))

                src_doc.append(sentence) #列表里面加入列表
                
                examples.append((src,seq_length,sent_length,tgt_doc1,tgt_doc2,src_doc)) 
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(list_of_examples):#进行pad 把你不够长的句子后面用pad来替代，然后转换成tensor的格式
        """
        This function defines how to organize a list of examples into a batch.
        You should do padding here, and convert the lsit to tensors of pytorch.
        """

        srcs = [x for x, y, z, a, b, c in list_of_examples]
        seq_lengths = [y for x, y, z, a, b, c in list_of_examples]
        sent_lengths = [z for x, y, z, a, b, c in list_of_examples]
        tgt_doc1 = [a for x, y, z, a, b, c in list_of_examples]
        tgt_doc2 = [b for x, y, z, a, b, c in list_of_examples]
        src_doc = [c for x, y, z, a, b, c in list_of_examples]

        #进行pad处理
        #1.得到最长的句子，和最长的文档大小
        max_word_num = find_martrix_max_value(seq_lengths)
        max_sent_num = max(sent_lengths)
        #2对srcs和scores、seg进行处理
        srcs = pad_srcs(srcs,max_word_num,max_sent_num)
        seq_lengths = pad_seq(seq_lengths,max_sent_num)


        # print(len(srcs))
        srcs = torch.LongTensor(srcs)
        seq_lengths = torch.LongTensor(seq_lengths)
        sent_lengths = torch.LongTensor(sent_lengths)


        srcs = Variable(srcs)
        seq_lengths = Variable(seq_lengths)
        sent_lengths = Variable(sent_lengths)

        
        return {"src": srcs,"seq_length":seq_lengths, "sent_length":sent_lengths,"tgt_doc1": tgt_doc1,"tgt_doc2": tgt_doc2,"src_doc":src_doc}
