from tqdm import tqdm

class Vocab(object): #为这个数据集建立词汇表
    def __init__(self,file_path):
        self._word2id = {} #这个是个字典
        self._id2word = []
        self.build(file_path)

    def __len__(self):
        return len(self._id2word)

    def word2id(self,word): 
        if word not in self._word2id:#如果这个词不在_word2id中，就表示不认识，返回不认识标记
            return self._word2id["<unk>"]
        return self._word2id[word]

    def id2word(self,id):
        if id >= len(self._id2word):
            return "<unk>"
        return self._id2word[id]

    def build(self,file_path):
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                idx = int(line.split()[0])
                word = line.split()[1]
                self._word2id[word] = idx #增加到里dict面
                self._id2word.append(word) #增加单词进去