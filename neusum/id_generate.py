import torch
'''
生成两个表格
'''
file_glove = './data/glove.6B.50d.txt'
file_src = './data/train_1000.txt.src'

file_vocab = './data/id_word.txt'
file_vector = './data/id_vector.txt'

glove_dict = {} #保存glove的内容的字典
with open(file_glove,'r',encoding='utf-8') as f: #读取一行就得到一个单词和一个词向量
    for line in f:
        word = line.split()[0]
        word_vector = line.split()[1:]
        glove_dict[word] = word_vector #把所有的单词加入到里面
print('---glove处理完毕--')

vocab = set() #set的内容
with open(file_src,'r',encoding='utf-8') as f:
    for line in f:
        sentence = line.split('##SENT##') #分成句子
        for sent in sentence: #对于每一个句子进行分词处理
            word_list = sent.split() #按照空格进行分割
            for word in word_list:
                vocab.add(word) #把单词加入进去
print('---vocab处理完毕---')

#先对特殊标志进行处理
with open(file_vocab,'a',encoding='utf-8')as fvocab,open(file_vector,'a',encoding='utf-8')as fvector:
    id = 0
    word = '<pad>'
    word_vector = [str(0) for i in range(50)]
    fvocab.write(str(id)+' '+word+'\n')
    fvector.write(str(id)+' '+' '.join(word_vector)+'\n')
    id = 1
    word = '<unk>'
    word_vector = torch.randn(50).tolist()
    word_vector = [str(v) for v in word_vector]
    fvocab.write(str(id)+' '+word+'\n')
    fvector.write(str(id)+' '+' '.join(word_vector)+'\n')
    id = 2
    word = '<sos>'
    word_vector = torch.randn(50).tolist()
    word_vector = [str(v) for v in word_vector]
    fvocab.write(str(id)+' '+word+'\n')
    fvector.write(str(id)+' '+' '.join(word_vector)+'\n')
    id = 3
    word = '<eos>'
    word_vector = torch.randn(50).tolist()
    word_vector = [str(v) for v in word_vector]
    fvocab.write(str(id)+' '+word+'\n')
    fvector.write(str(id)+' '+' '.join(word_vector)+'\n')
    fvector.close()
    fvocab.close()

for word in vocab:#遍历set中的单词
    id = id + 1
    if word in glove_dict:
        word_vector = glove_dict[word]
    else:
        word_vector = torch.randn(50).tolist()
        word_vector = [str(v) for v in word_vector]
    with open(file_vocab,'a',encoding='utf-8')as fvocab,open(file_vector,'a',encoding='utf-8')as fvector:
        fvocab.write(str(id)+' '+word+'\n')
        fvector.write(str(id)+' '+' '.join(word_vector)+'\n')

print(id) #31064

    
