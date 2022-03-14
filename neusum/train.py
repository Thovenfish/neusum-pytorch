import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from vocab import Vocab
from dataset import MyDataset
from myAdam import MyAdam
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import logging


'''
增加一个step的mask操作
屏蔽了计算loss的三个句子...不太明白这里的操作：因为我是觉得已经经过mask,并且通过softmax编程趋近于0，对结果没有影响
'''
def gen_judge_tgt(tgt):
    mask = torch.ByteTensor(tgt.shape[0], tgt.shape[1]).zero_().to(device)
    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            if tgt[i][j]!=-1:
                mask[i][j]=1
    mask = mask.float()
    return mask  

def regression_loss(pred_scores, gold_scores, mask, judge, crit):
    """
    :param pred_scores: (step, batch, doc_len)
    :param gold_scores: (batch*step, doc_len) #这些的格式要看?
    :param mask: (batch, doc_len)
    :param crit:
    :return:
    """
    pred_scores = pred_scores.transpose(0, 1).contiguous()  # (batch, step, sent_len)
    if isinstance(crit, nn.KLDivLoss):
        # TODO: we better use log_softmax(), not log() here. log_softmax() is more numerical stable.
        ls = nn.LogSoftmax(2)
        pred_scores = ls(pred_scores) #先弄来成0？可是经过softmax本来就是0了?2
    pred_scores = pred_scores * (1 - mask).unsqueeze(1).expand_as(pred_scores) #把句子部分mask掉
    loss = crit(pred_scores, gold_scores)
    # loss = loss * (1 - mask).unsqueeze(1).expand_as(loss) #我感觉这一步不是很有必要，因为上面已经处理过了？3
    loss = loss*judge.unsqueeze(-1).expand_as(loss) #进行处理
    reduce_loss = loss.sum() #总和的损失
    # reduce_loss = loss.sum()/(pred_scores.shape[0]) #分均的损失
    return reduce_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocb_size = 0
emd_size = 50
hidden_size = 128
input_size = 256
hidden_size_de = 256
max_step = 6
hq = 256
hd = 256
hs = 256
data_use = 'alldata'
model_save = './model/'+data_use+'/' #保存模型的位置
file_src = './data/'+data_use+'/train.src' #源文件
file_tgt = './data/'+data_use+'/train_tgt_score_soft.txt' #tgt经过soft处理的文件
file_vocab = './data/'+data_use+'/id_word.txt'
file_vector = './data/'+data_use+'/id_vector.txt'
log_file = './logger/'+data_use+'/3.txt'
epochs = 20
print_batch = 100
save_epoch =10

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")


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

regression_crit = nn.KLDivLoss(size_average=False, reduce=False) #参数不知道？
optimizer = MyAdam(model.parameters(),lr=0.001) #优化器
model.to(device)

#1.读取词嵌入，初始化embedding
model.sent_encoder.embedding.weight.data.copy_(torch.from_numpy(vector_metrx))
model.sent_encoder.embedding.weight.requires_grad = False #不去更新参数
print("---词嵌入加载完毕---")

#2.dataloader
train_dataset = MyDataset(file_src, file_tgt, src_vocab) #对数据进行编码
#数据集加载
train_dataloader = DataLoader( #框架中自己的函数
        train_dataset,
        batch_size=64, #每64个句子组成一个batch
        shuffle=True, #打乱顺序
        collate_fn=MyDataset.collate_fn,
        num_workers=4, #是否使用多线程
    )
model.train() #model.trainning = True
for epoch in range(epochs):
    loss = 0
    reg_loss = 0
    for i, batch in enumerate(train_dataloader):
        src = batch['src'].to(device)
        seq_length = batch['seq_length'].to(device)
        sent_length = batch['sent_length'].to(device)
        tgt = batch['tgt'].to(device)
        gold_score = batch['score'].to(device)
        batch_len = src.shape[0]
        judge = gen_judge_tgt(tgt)

        # model.zero_grad()
        optimizer.zero_grad()  
        All_score, mask = model(src,seq_length,sent_length,tgt)
       
        loss = regression_loss(All_score, gold_score, mask,judge,regression_crit)
        # loss = reg_loss
        # loss.requires_grad = True
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5,norm_type=2) #?[-5,5]这个怎么弄
        optimizer.step()
        reg_loss =reg_loss + loss
        if i% print_batch ==0:
            print_str = str(i)+"/"+str(epoch+1)+" : loss = "+str(loss.data)+" , avg_loss = "+str(loss/batch_len)
            print(print_str)
            logger.info(print_str)
    epoch_str = "-----epoch:"+str(epoch+1)+" avg_loss:"+str(reg_loss/(train_dataset.__len__()))
    print(epoch_str)
    logger.info(epoch_str)
    print()
    logger.info("\n")

    if (epoch+1)%save_epoch == 0:
        model_path = model_save + "model-epoch-"+str(epoch+1)+".ckpt" #每20个保存一次
        torch.save(model.state_dict(), model_path)
