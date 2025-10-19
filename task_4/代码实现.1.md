## 下一词预测
 ```
 import nltk  
from nltk.corpus import brown  
from collections import Counter, defaultdict  
import re  
  
#1. 下载一次即可  
nltk.download('brown')  
  
#2. 读语料 & 转小写  
words = [w.lower() for sent in brown.sents() for w in sent]  
  
#3. 建表  
uni = Counter(words)  
bi  = Counter(zip(words, words[1:]))  
tri = Counter(zip(words, words[1:], words[2:]))  
  
V = len(uni)  
k = 1  # Add-k 平滑  
  
#4. trigram 预测  
def predict_next(sentence: str, top_k=5):  
    # 简单分词（空格+去标点）  
    tokens = re.findall(r"\b\w+\b", sentence.lower())  
    if len(tokens) < 2:  
        tokens = ["<s>"] + tokens      # 不足两词补句首  
    w1, w2 = tokens[-2], tokens[-1]  
  
    candidates = defaultdict(float)  
    for (c1, c2, c3), cnt in tri.items():  
        if c1 == w1 and c2 == w2:  
            numer = cnt + k  
            denom = bi[(w1, w2)] + k * V  
            candidates[c3] = numer / denom  
  
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]  
#5. 交互测试  
if __name__ == "__main__":  
    while True:  
        sent = input("\n输入句子（空行退出）：").strip()  
        if not sent:  
            break  
        for w, p in predict_next(sent):  
            print(f"  {w}  P={p:.4f}")
  ```




  
## 相似词查询
 ```
 import nltk
from gensim.models import Word2Vec
from nltk.corpus import brown
 #若第一次使用 brown，先下载
nltk.download('brown')

#训练（约几十秒）
sents = brown.sents()          # 已按句子切分好的列表
model = Word2Vec(
        sentences=sents,
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,                # 1=Skip-gram
        epochs=5,
        workers=4)

#相似词查询函数
def similar(word, topn=5):
    """
    word : 待查询的词
    topn : 返回最相似的词个数
    """
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return f"'{word}' 不在词汇表中"

#交互式使用
if __name__ == "__main__":
    while True:
        w = input("请输入一个词 (q 退出)：").strip().lower()
        if w == 'q':
            break
        print(similar(w))
```


   


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NjUwMzEwODZdfQ==
-->