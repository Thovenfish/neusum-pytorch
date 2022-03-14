import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from vocab import Vocab
from dataset_test import MyDatasetTest
from myAdam import MyAdam
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PyRouge.Rouge.Rouge import Rouge as r1
from rouge import Rouge as r2

'''
1.什么时候停止？目前采取6步+set的方式。
2.为什么每一次运行测试的结果不一样？好像其中没有随机的内容？保存模型的问题吗？
3.进行tgt的分数的文档是新产生的文档还是以前的文档？
4.rouge分数是百分数吗？可以直接使用这个包里面的rouge进行计算吗？
small - test:300-0 train 1000-6 小数据很容易过拟合了，所以不用训练很多次，20次分数20，但是后面的都只有16左右了
alldata 
对于新的内容 需要生成关于词向量的两个文件，train的一个文件，test的一个文件
（如果使用的是自己抽取的tgt进行分数计算，那么还需要重新生成新的文件-直接抽取形成即可简单的，那么不需要使用test_tgt_doc.txt文件了）
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rouge = r1(use_ngram_buf=True)
rouge_sim = r2()

vocb_size = 0
emd_size = 50
hidden_size = 128
input_size = 256
hidden_size_de = 256
max_step = 3
hq = 256
hd = 256
hs = 256
data_use = 'alldata' #alldata
model_loadpath = './model/'+data_use+'/model-epoch-8.ckpt' #保存模型的位置
# model_loadpath = './model/smalldata/model-epoch-20.ckpt' #保存模型的位置
test_src = './data/'+data_use+'/test.src' #源文档
test_tgt = './data/'+data_use+'/test_tgt_score_soft.txt' #进行soft处理之后的分数
test_doc_tgt = './data/'+data_use+'/test_tgt_doc.txt' #tgt的文档
file_vocab = './data/'+data_use+'/id_word.txt'
file_vector = './data/'+data_use+'/id_vector.txt'
# file_vocab = './data/smalldata/id_word.txt'
# file_vector = './data/smalldata/id_vector.txt'


#读取id2word到vocab
src_vocab = Vocab(file_vocab)
print("-----单词表建立完成，词汇大小：",src_vocab.__len__(),"-----")
vocb_size = src_vocab.__len__()


#读取词向量矩阵
vector_metrx = np.random.uniform(-1,1,(vocb_size,emd_size)) #建立一个矩阵
with open(file_vector,encoding="utf-8") as f:
    for line in f:
        id_index = int(line.split()[0])
        vecter_txt = line.split()[1:]
        vector_metrx[id_index] =list(map(eval, vecter_txt))
print("---权重矩阵计算完毕---")


sent_encoder = model.Encoder(vocb_size,emd_size,hidden_size)
doc_encoder = model.DocumentEncoder(input_size,hidden_size)
decinit = model.DecInit(hidden_size,hidden_size_de)  #128 256
select = model.Select(input_size,hidden_size_de,max_step,hq,hd,hs) #256 256 5 256 256 256
model = model.NeuSum(sent_encoder,doc_encoder,decinit,select)

model.to(device)
model.load_state_dict(torch.load(model_loadpath,map_location=torch.device('cpu')))
#1.读取词嵌入，初始化embedding
model.sent_encoder.embedding.weight.data.copy_(torch.from_numpy(vector_metrx))
# model.sent_encoder.embedding.weight.requires_grad = False #不去更新参数
print("---词嵌入加载完毕---")


#2.dataloader
test_dataset = MyDatasetTest(test_src, test_tgt, src_vocab, test_doc_tgt) #对数据进行编码
#数据集加载
test_dataloader = DataLoader( #框架中自己的函数
        test_dataset,
        batch_size=1, #每64个句子组成一个batch
        shuffle=False, #打乱顺序,True的话会出错
        collate_fn=MyDatasetTest.collate_fn,
        num_workers=0, #是否使用多线程
    )

# model.train(False) #不能使用model.training = False
model.eval()
sum_score_sim_1 = 0 
sum_score_sim_2 = 0 
sum_score_sim_l = 0 
sum_score_21 = 0
sum_score_22 = 0
i = 0
for i, batch in enumerate(test_dataloader): #如果是非有效的，则在进行dataset处理的时候就已经消除掉了,所以有效个数就是i值
    src = batch['src'].to(device)
    seq_length = batch['seq_length'].to(device)
    sent_length = batch['sent_length'].to(device)
    sentence = batch['src_doc'][0][0] #一个列表的列表
    gold1 = batch['tgt_doc1'][0][0] #
    gold2 =batch['tgt_doc2'][0][0] #原文

    All_score, mask = model(src,seq_length,sent_length)

    # print(All_score.shape) #step 1 sent
    score = All_score.squeeze() #step sent
    #---------------------------old
    max_idx=[]
    for j in range(max_step):
        max_score, m_ids = score.max(dim=1)
        m_id = m_ids[j]
        max_idx.append(m_id)
    # max_score, max_idx = score.max(dim=1)
        # 进行mask
        # if j <(max_step-1):
        #     for k in range(j+1,max_step):
        #         score[k][m_id] = -1e8
    #---------------------------------new
    # max_idx=[]
    # for j in range(max_step):
    #     max_score, m_ids = score.max(dim=1)
    #     m_id = m_ids[j]
    #     if j!=0 and m_id == m_ids[j-1]:
    #         break
    #     max_idx.append(m_id)
    #---------------------------------
    result = ' '.join([sentence[x] for x in max_idx]) #形成句子

    score_test = rouge.compute_rouge([gold2],[result])['rouge-1']['r'][0]
    sum_score_21 = sum_score_21 + score_test #
    score_test = rouge.compute_rouge([gold2],[result])['rouge-2']['r'][0]
    sum_score_22 = sum_score_22 + score_test #
    
    score_test = rouge_sim.get_scores(gold2,result)[0]['rouge-1']['r']
    sum_score_sim_1 = sum_score_sim_1 +score_test

    score_test = rouge_sim.get_scores(gold2,result)[0]['rouge-2']['r']
    sum_score_sim_2 = sum_score_sim_2 +score_test

    score_test = rouge_sim.get_scores(gold2,result)[0]['rouge-l']['r']
    sum_score_sim_l = sum_score_sim_l +score_test

    # break

    if i%200==0:
        print(i)

avg_score = sum_score_21/float(i) #原文-1
print(avg_score)
avg_score = sum_score_22/float(i) #原文-2
print(avg_score)

avg_score = sum_score_sim_1/float(i) #原文-2
print(avg_score)
avg_score = sum_score_sim_2/float(i) #原文-2
print(avg_score)
avg_score = sum_score_sim_l/float(i) #原文-2
print(avg_score)