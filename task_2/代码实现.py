### 用python实现点乘注意力机制（自注意力，交叉注意力）
 ```
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#1. 单头点乘 Self-Attention
class DotProductSelfAttention(nn.Module):
    """
    输入: x  (batch, seq_len, d_model)
    输出: out  (batch, seq_len, d_model)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale   = 1.0 / math.sqrt(d_model)

        # 把 Q/K/V 的线性映射做成一个矩阵乘法，更高效
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)                # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)        # (B, L, d_model) * 3

        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # (B, L, L)
        if mask is not None:                  # mask: (B, L, L) 或 (L, L)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)              # (B, L, d_model)
        return self.out_proj(out)

#2. 单头 Cross-Attention
#Encoder-Decoder Attention 的经典用法
class DotProductCrossAttention(nn.Module):
    """
    query : (B, Lq, d_model)
    key   : (B, Lk, d_model)  (来自另一个序列)
    value : (B, Lk, d_model)  通常 key == value
    输出  : (B, Lq, d_model)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale   = 1.0 / math.sqrt(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        q = self.q_proj(query)                # (B, Lq, d_model)
        k = self.k_proj(key)                  # (B, Lk, d_model)
        v = self.v_proj(value)                # (B, Lk, d_model)

        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # (B, Lq, Lk)
        if mask is not None:                  # mask: (B, Lq, Lk)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.bmm(attn, v)              # (B, Lq, d_model)
        return self.out_proj(out)

#3. 多头版本：Multi-Head Self / Cross Attention
class MultiHeadAttention(nn.Module):
    """
    统一实现 Self-Attention 与 Cross-Attention：
      如果 key == value == x 则为 Self-Attention
      否则为 Cross-Attention
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None):
        """
        query: (B, Lq, d_model)
        key  : (B, Lk, d_model)  若 None 则使用 query
        value: (B, Lk, d_model)  若 None 则使用 query
        mask : (B, Lq, Lk)  或 None
        """
        if key is None and value is None:
            key = value = query           # Self-Attention
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 拼接 QKV 进行一次投影，然后拆分
        if key is query and value is query:
            qkv = self.qkv_proj(query)    # (B, L, 3*d_model)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # 独立投影
            q = self.qkv_proj.weight[:self.d_model] @ query.transpose(-2, -1)
            k = self.qkv_proj.weight[self.d_model:2*self.d_model] @ key.transpose(-2, -1)
            v = self.qkv_proj.weight[2*self.d_model:] @ value.transpose(-2, -1)
            q, k, v = [x.transpose(-2, -1) for x in (q, k, v)]

        # reshape 为 (B * n_heads, L, d_k)
        def reshape(x):
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2).contiguous().view(B * self.n_heads, -1, self.d_k)

        q, k, v = map(reshape, (q, k, v))

        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # (B*n_heads, Lq, Lk)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(B * self.n_heads, Lq, Lk)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.bmm(attn, v)                                   # (B*n_heads, Lq, d_k)
        out = out.view(B, self.n_heads, Lq, self.d_k).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)
#小测试：Self & Cross Attention
if __name__ == "__main__":
    B, L, d = 2, 5, 64
    x = torch.randn(B, L, d)
    y = torch.randn(B, L + 3, d)

    self_attn  = MultiHeadAttention(d, n_heads=8)
    cross_attn = MultiHeadAttention(d, n_heads=8)

    out_self  = self_attn(x)        # (B, L, d)
    out_cross = cross_attn(x, y, y) # (B, L, d)

    print("Self  output:", out_self.shape)
    print("Cross output:", out_cross.shape)
```
### 用Python实现贪心搜索和束搜索
```
import math  
  
# 模拟一个简单的文本生成场景  
# 假设我们有一个简单的词汇表：["我","爱","编程","<eos>"]  
# 其中<eos>是结束符  
  
vocab=["我","爱","编程","<eos>"]# 定义一个python数组作为可选词汇表  
vocab_size=len(vocab)  
  
# 模拟神经网络预测的概率分布  
#（在实际中这些概率来自模型）  
def get_next_word_probs(previous_words):  
    """根据前面的词预测下一个词的概率"""  
    if previous_words==[]:  
        return [0.6,0.3,0.08,0.02]  
    # 句子开头，“我”的概率最高  
    elif previous_words==["我"]:  
        return [0.1,0.7,0.15,0.05]  
    # “我”后面“爱”的概率最高  
    elif previous_words==["我","爱"]:  
        return [0.05,0.1,0.8,0.05]  
    # “爱”后面“编程”概率最高  
    elif previous_words==["我","爱","编程"]:  
        return [0.01,0.01,0.08,0.9]  
    # “编程后面”结束符概率最高  
    else:  
        return [0.25,0.25,0.25,0.25]  
    # 句子已经结束，返回相等概率的默认值  
  
def greedy_search():  
    """贪心搜索：每步选择概率最大的值"""  
    print("====贪心搜索===")  
  
    sequence=[]# 生成的序列  
    total_prob=1.0# 总概率为1  
  
    for step in range(5):# for 变量 in range(次数)  
        # 获取下一个词的概率  
        probs=get_next_word_probs(sequence)  
  
        # 选择概率最大的词  
        best_idx=probs.index(max(probs))# max(probs)返回probs数组的最大值,而probs.index(最大值)返回这个值在数组中的下标  
        best_word=vocab[best_idx]# 这是概率最大的那个词  
        best_prob=probs[best_idx]# 这是最大的概率值  
  
        # 添加到序列中  
        sequence.append(best_word)# 列表.append（要添加的元素），用于在列表的末尾添加一个新元素  
        total_prob *=best_prob# 累乘计算句子总概率  
  
        print(f"步骤{step+1}:选择‘{best_word}’(概率：{best_prob:.3f})")  
        print(f"当前序列：{sequence},总概率：{total_prob:.6f}")  
  
        # 如果遇到结束符，停止生成  
        if best_word=="<eos>":  
            break  
  
    return sequence,total_prob  
  
def beam_search(beam_width=2):# 束宽度为2  
    """束搜索，每次保留多个候选序列"""  
    print("===束搜索===")  
  
    # 初始：空序列，概率为1  
    beams=[([],1.0)]# 每个元素是（序列，概率），序列就是可能出现的句子  
  
    for step in range(5):# 最多5步  
        print(f"\n步骤{step+1}:")  
        candidates=[]# 所有候选序列  
  
        # 对每个候选序列进行扩展  
        for seq,prob in beams:  
            # 如果序列以<eos>结尾，不再扩展，直接添加到序列中  
            if len(seq)>0 and seq[-1]=="<eos>":# seq[-1]提取序列的最后一个词  
                candidates.append((seq,prob))  
                continue  
            # 获取下一个词的概率（数组）  
            next_probs=get_next_word_probs(seq)  
            # 考虑所有可能的扩展  
            for i,word_prob in enumerate(next_probs):# enumerate函数，在遍历概率序列的同时得到元素的索引和值  
                new_seq=seq+[vocab[i]]# 添加新词  
                new_prob=prob*word_prob# 更新概率  
                candidates.append((new_seq,new_prob))  
  
                # 按照概率排序，选择前面最好的beam_width个  
                candidates.sort(key=lambda x:x[1],reverse=True)  
                # lambda用于创建简单的匿名函数，这里用于返回每个序列的第二个元素  
                # sort按照概率从高到低把candidates排序  
                beams=candidates[:beam_width]  
  
            # 显示当前保留的候选  
            print(f"保留的{beam_width}个候选：")  
            for i,(seq,prob) in enumerate(beams):  
                print(f"候选{i+1}:{seq},概率：{prob:.6f}")  
  
            # 如果所有候选都以<eos>结尾，提前结束  
            if all(len(seq)>0 and seq[-1]=="<eos>"for seq in beams):  
                break  
  
    # 返回最好的序列  
    best_sequence,best_prob=beams[0]  
    return best_sequence,best_prob  
  
# 运行实例  
# 贪心搜索  
greedy_seq,greedy_prob=greedy_search()  
print(f"\n贪心搜索结果：{greedy_seq},概率：{greedy_prob:.6f}")  
# 束搜索  
beam_seq,beam_prob=beam_search(beam_width=2)  
print(f"\n束搜索结果：{beam_seq},概率：{beam_prob:.6f}")  
# 比较两种方法  
if beam_prob>greedy_prob:  
    print(f"束搜索找到了概率更高的序列")  
elif beam_prob<greedy_prob:  
    print("贪心搜索找到了概率更高的序列")  
else:  
    print(f"概率相等")
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbOTgyNTc1OTk4XX0=
-->
