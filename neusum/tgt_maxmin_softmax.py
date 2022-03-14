import torch
import torch.nn as nn
import torch.nn.functional as F
'''
生成经过maxmin和softmax的值
'''
T = 1

def idx_deal(idx_list):
    y = []
    for k in idx_list:
        if k == '':
            continue
        y.append(k.strip())
    string = ' '.join(y)
    return string

def score_deal(score):
    score_list = [float(s) for s in score]
    score_tensor = torch.FloatTensor(score_list) #转换成tensor
    # socre_tensor = F.normalize(socre_tensor)
    max_s = score_tensor.max()
    min_s = score_tensor.min()
    score_tensor = (score_tensor-min_s)/(max_s-min_s)*T #norm
    score_tensor = torch.softmax(score_tensor,dim = -1) #softmax
    score_tensor = score_tensor.tolist()
    score_tensor = [str(s) for s in score_tensor]
    return ' '.join(score_tensor)

def score_gen(tgtnum,sentlen):
    score_sig = [str(0) for i in range(sent_len)] #产生这个
    score_sig_str = ' '.join(score_sig)
    score_mul = ''
    for j in range(tgtnum):
        score_mul = score_mul + '\t'+score_sig_str
    return score_mul


# file_path = './data/tgt_change_1000.txt'
# file_path_d = './data/tgt_change.txt'
file_path = './data/smalldata/test_tgt_score.txt'
file_path_d = './data/smalldata/test_tgt_score_soft.txt'
none_i = 0
with open(file_path,'r',encoding='utf-8') as f:
    # line  = f.readlines()
    # for i in range(1001):
    for line in f:
        line_list = line.split('\t')
        if line_list[0]=='None':
            with open(file_path_d,'a',encoding='utf-8') as fd:
                fd.write('None\t0\n') #就保持原样就可以
            none_i = none_i+1
            continue
        tgt_idx = line_list[0]
        idx_list = tgt_idx.split('(')[1].split(')')[0].split(',')
        idx_str = idx_deal(idx_list)
        sent_len = 0  #句子数目
        score_str_s = ''
        for j in range(1,len(line_list)):
            score = line_list[j].split()
            sent_len = len(score) 
            score_str_s = score_str_s + '\t'+score_deal(score)
        # print("---",sent_len)
        # print("---",6-len(line_list))
        score_str_e = score_gen((7-len(line_list)),sent_len) #分数扩展成6个
        with open(file_path_d,'a',encoding='utf-8') as fd:
            fd.write(idx_str+score_str_s+score_str_e+'\n')
        # print(idx_str+score_str_s+score_str_e+'\n')

print(none_i) #输出为none的数量