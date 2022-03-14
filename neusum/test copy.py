import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn as nn
import torch.nn.functional as F

#测试转置
# a = torch.rand(2,8,3) #->(2,4,6)
# b = a.transpose(0,1)
# c = b.reshape(8,2*3) #8,6
# d = b.reshape(4,2,6)
# e = b.reshape(2,4,6)
# f = d.transpose(0,1)
# # print(a)
# # print(b)
# # print(c)
# # print(d)
# # print(e)
# print(f.shape) #f = a.transtpose(0,1).reshape(4,2,6).transpose(0.1)

# a = torch.rand(2,8,3)
# b = a.unsqueeze(0)
# # print(b.s #resize是不会改变原本的
# print(a.shape) 
# with open('./cnndm_acl18-master/cnn_dailymail/out-rouge.txt','r',encoding='utf-8') as f1:
#     s = f1.readline()
# with open('./cnndm_acl18-master/cnn_dailymail/train.txt.src','r',encoding='utf-8') as f2:
#     t = f2.readline()
# t_len = len(t.split("##SENT##"))
# s_len = len(s.split('\t')[1].split()) #45 45
# print(t_len,s_len)
# # print(s.split('\t')[0]) Q的大小是：tgt_len * src_len :tgt的数目*源文档句子数目
# # print(s.split('\t')[1])
# # print(s.split('\t')[2])

# -------------------
# a = torch.FloatTensor([[1,2,3,4],[1,0,0,0],[3,1,4,0]]).resize_(3,4,1) #batch在前
# length = [4,1,3]
# p_a = pack(a,length,batch_first=True,enforce_sorted=False)
# print(p_a.data) # 132103
# b = torch.FloatTensor([[1,2,3,4],[3,1,4,0],[1,0,0,0]]).resize_(3,4,1).transpose(0,1) #如果没有batch_first 就是seqlen在前
# # print(b)
# length = [4,3,1]
# p_b = pack(b,length)
# print(p_b.data)#123311

# rnn = nn.GRU(1,2,bidirectional=True) #相当于word_embedding_size=1 
# rnn_a = nn.GRU(1,2,batch_first=True,bidirectional=True)
# out,hidden = rnn_a(p_a) 
# # rnn = nn.GRU(1,2,bidirectional=True)
# # out_b,hidden_b = rnn(p_b)


# print("out",out)
# # print("out_b",out_b)


# up_out = unpack(out) #返回的是tuple
# print(up_out[0].shape) #b s h

# # up_out_b = unpack(out_b)
# # print(up_out_b[0].shape)# s b h

# print(up_out[0])
# # print(up_out_b[0])


# a = torch.FloatTensor([[1,2,3,4],[1,0,0,0],[3,1,4,0]]).resize_(3,4,1) #batch在前
# length = [4,1,3]
# p_a = pack(a,length,batch_first=True,enforce_sorted=False)#batch在前
# print(p_a.data) # 132103

# rnn_a = nn.GRU(1,2,batch_first=True,bidirectional=True)#batch在前
# out,hidden = rnn_a(p_a) 
# print("out",out)
# up_out = unpack(out,batch_first=True) #返回的是tuple #batch在前
# print(up_out[0].shape) #b s h
# print(up_out[0])

# m = [(2,3),(1,2),(3,3),(0,8),(2,9)]
# k = [v for x,v in m]
# print(k)

# c = torch.FloatTensor([[1,2,3],[2,3,6]]) #[1.0, 2.0, 3.0, 2.0, 3.0, 6.0]
# c = torch.rand(2,3,4)
# b = c.view(-1,c.size(2))
# print(b.shape)

# c = torch.FloatTensor([[[1,0,0],[2,0,0],[3,1,2]],[[1,0,6],[0,0,0],[0,0,0]]]) #2,3,3
# b = [3,1]
# k = pack(c,b,batch_first=True,enforce_sorted=False)
# u = unpack(k,batch_first=True)
# print(k.data)
# print(u[0])

# c = torch.rand(1,3,4)
# # print(c)
# # print(c[:,1,:])
# # print(c[:,1,:].shape)
# lay = nn.Linear(4,2)
# out = lay(c)
# print(out.shape)

#--------------处理tgt标号-------------
# p = []
# with open('./cnndm_acl18-master/cnn_dailymail/syt.txt') as f:
#     for line in f:
#         # print(line.strip())
#         y = []
#         x = line.split('(')[1].split(')')[0].split(',')
#         for k in x:
#             if k == '':
#                 continue
#             y.append(int(k.strip()))
#         p.append(y)
# print(p)
# # torch.LongTensor(p) #不能转换成tensor因为不一样的维度

# c = torch.FloatTensor([[1,2,3],[2,3,6],[0,0,0]]).unsqueeze(-1) #3,3,1
# rnn = nn.GRU(1,2,batch_first=True,bias=False)
# # print(c.shape)
# b,_= rnn(c)
# print(b.shape) #3,3,2
# print(b)

# tgt_idx = [9,2,3,-1,-1]
# t_judge = [False if x == -1 else True for x in tgt_idx]
# print(t_judge)


#测试kl散度计算loss值
# cite = nn.KLDivLoss(size_average=False, reduce=False) #参数不知道？

# godl_score = torch.Tensor([[9,2,3,0,0],[9,3,3,0,0],[0,0,0,0,0]])
# score = torch.Tensor([[2,3,6,0,0],[1,2,4,0,0],[0,0,0,0,0]])


# print(cite(score,godl_score))
# print(score.sum())

# godl_score = torch.Tensor([[9,2,3],[9,3,3]])
# score = torch.Tensor([[2,3,6],[1,2,4]])
# print(cite(score,godl_score))
# print(score.sum())

# godl_score = torch.FloatTensor([[9,2,3,0,0],[9,3,3,1,0],[2,1,-1,4,0]])
# t_judge = torch.Tensor([0,0,-1]).unsqueeze(1)
# godl_score = godl_score + godl_score * t_judge.expand_as(godl_score)
# print(godl_score)
# print(t_judge.expand_as(godl_score))

# godl_score = torch.FloatTensor([[9,2,3,0,0],[9,3,3,1,2],[2,1,-1,4,-1e8]])
# print(F.softmax(godl_score,dim=1))

'''
tensor([[9.9638e-01, 9.0858e-04, 2.4698e-03, 1.2296e-04, 1.2296e-04],
        [9.9383e-01, 2.4635e-03, 2.4635e-03, 3.3339e-04, 9.0626e-04],
        [1.1355e-01, 4.1773e-02, 5.6533e-03, 8.3902e-01, 0.0000e+00]]) #可以看到这里很小到0 是按照行来计算的，正确
'''
# cite = nn.KLDivLoss(reduction='none') #参数不知道？

# godl_score = torch.Tensor([[9,2,3,-1e8,-1e8],[9,3,3,-1e8,-1e8],[1,2,-1e8,-1e8,-1e8]])
# godl_score = F.softmax(godl_score,dim=1)
# t_judge = torch.Tensor([1,1,0]).unsqueeze(1)
# godl_score = godl_score * t_judge.expand_as(godl_score)
# print(godl_score)
# score = torch.Tensor([[2,3,6,0,0],[1,2,4,0,0],[0,0,0,0,0]])
# s =  cite(score,godl_score)
# print(s)
# print(s.sum())

# godl_score = torch.Tensor([[9,2,3],[9,3,3]])
# score = torch.Tensor([[2,3,6],[1,2,4]])
# godl_score = F.softmax(godl_score,dim=1)
# print(godl_score)
# s =  cite(score,godl_score)
# print(s)
# print(s.sum())

# print(cite(torch.Tensor([0,0]),torch.Tensor([1,0])))
# m = []
# c = torch.Tensor([[9,2,3],[9,3,3]])
# b = torch.Tensor([[9,2,3],[9,3,3]])
# m += [c]
# m = torch.stack(m) #1 2 3
# # print(m)

# gru = nn.GRUCell(3,3)
# out = gru(c,m) #2,3
# print(out.shape)

# v = set()
# v.add('a')
# v.add('b')
# print(v)
# for c in v:
#         print(c)

# x = [str(0) for i in range(50)]
# x = torch.randn(50).tolist()
# x = [str(i) for i in x]
# print(' '.join(x))

# m = []
# x = torch.randn(5,3)
# y = torch.randn(5,3)
# m.append(x)
# m.append(y)
# print(x)
# print(y)
# k = torch.stack(m)
# print(k.shape)
# print(k)

# godl_score = torch.Tensor([[9,2,3,-1e8,-1e8],[9,3,3,-1e8,-1e8],[1,1,1,1,1]])
# s = nn.LogSoftmax(1)
# m = s(godl_score)
# print(m)

# x = torch.randn(5,3,4)

# y = torch.tensor([[1,1,0],[1,0,0],[1,0,0],[1,1,0],[1,0,0]])

# print(x)
# pred_scores * (1 - mask).unsqueeze(1).expand_as(pred_scores)
# x = x * y.unsqueeze(-1).expand_as(x)
# print(x)

# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)
# handler = logging.FileHandler("log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
 
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# s = "llllll"
# formatter = logging.Formatter('%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info(s)
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")

