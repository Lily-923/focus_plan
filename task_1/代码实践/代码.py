## PyTorch神经网络
### 代码
``` 
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  
from torchvision import datasets  
from torchvision.transforms import ToTensor  

#1.数据准备部分  
#下载训练集  
training_data = datasets.FashionMNIST(  
    root="data",  
    train=True,  
    download=True,  
    transform=ToTensor(),  
)  
  
#下载测试集  
test_data = datasets.FashionMNIST(  
    root="data",  
    train=False,  
    download=True,  
    transform=ToTensor(),  
)  
batch_size = 64 # 分批次处理，每批处理的数据量  
  
#创建数据加载器  
train_dataloader = DataLoader(training_data, batch_size=batch_size)  
test_dataloader = DataLoader(test_data, batch_size=batch_size)  
  
  
#2.模型定义部分与参数初始化  
#选择设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using {device} device")  
#2.定义模型  
class NeuralNetwork(nn.Module):  
    def __init__(self):  
        super().__init__()  
        # 手动初始化权重和偏置  
        self.W1=nn.Parameter(torch.randn(28*28,512)*0.01)# 这是第一层的权重矩阵。0.01是权重初始化的缩放因子，避免权重太大或太小  
        # 权重矩阵的形状是（输入维度，输出维度），这里输入维度是28*28（图像的展平尺寸），输出维度是512（第一层的神经元数量）  
        self.b1=nn.Parameter(torch.zeros(512))  
        # 偏置向量，长度等于该层的输出维度，让神经元再没有输出的时候也能激活  
  
        self.W2=nn.Parameter(torch.randn(512,512)*0.01)  
        self.b2=nn.Parameter(torch.zeros(512))  
  
        self.W3=nn.Parameter(torch.randn(512,10)*0.01)  
        self.b3=nn.Parameter(torch.zeros(10))  
  
  
    def forward(self, x):  
         x_flat = x.view(-1, 28 * 28)# 展平输入  
  
        # 第一层：线性变换+ReLU激活  
         z1=x_flat@self.W1+self.b1# @代表矩阵乘法，输出=输入矩阵*权重矩阵+偏置项  
         a1=F.relu(z1)# ReLU激活函数，输入大于0，输出等于输入；输入小于0，输出为0，引入非线性  
        # 第二层：线性变换+ReLU激活  
         z2=a1@self.W2+self.b2  
         a2=F.relu(z2)  
        # 第三层(输出层）  
         z3=a2@self.W3+self.b3  
  
         return z3  
  
#3.手动实现前向传播  
def manual_forward(x,model):  
    x_flat=x.view(-1,28*28)  
  
    # 第一层  
    z1=x_flat@model.W1+model.b1# 使用模型中的已经初始化的参数矩阵和偏置项  
    a1=F.relu(z1)# 激活函数  
    # 第二层  
    z2=a1@model.W2+model.b2  
    a2=F.relu(z2)  
    # 输出层  
    z3=a2@model.W3+model.b3  
  
    return z3,(x_flat,z1,a1,z2,a2,z3)# 返回z3（输出结果），输入，还有各个参数  
  
#4.手动实现损失计算  
def manual_cross_entropy(logits,y):  
    # 计算softmax  
    exp_logits=torch.exp(logits-torch.max(logits,dim=1,keepdim=True)[0])# 数值稳定性  
    softmax=exp_logits/torch.sum(exp_logits,dim=1,keepdim=True)# 实现softmax函数，输出一系列概率分布，并且所有元素之和为1  
  
    # 计算交叉熵损失  
    n=y.shape[0] # 计算有多少个样本，y是正确答案的标签，y.shape[0]就是获取这批图片的数量  
    log_softmax=torch.log(softmax+1e-10)# softmax是模型预测的概率，+e-10是一个很小的数（为了防止概率为0时，log(0)无意义，torch.log是取对数）  
    loss=-torch.sum(log_softmax[range(n),y])/n# 计算n个样本的平均损失  
  
    return loss,softmax# 最终返回loss和softmax概率  
  
#5.手动实现反向传播  
def manual_backward(x,y,logits,softmax,cache,model):  
    x_flat,z1,a1,z2,a2,z3=cache# 从一个叫cache的变量中取出之前保存的值  
    n=y.shape[0]# 获取样本数量  
  
    # 计算损失函数对于第三层输出的梯度  
    grad_z3=softmax.clone()# 复制一份softmax旳张量，并且保留原本的softmax  
    grad_z3[range(n),y]=-1# range(n),y定位到了每个样本的真实类别所对应的那个预测概率，再减去1  
    # 符合公式：梯度=预测概率-真实标签  
    grad_z3/=n# 将总梯度转化成平均梯度  
  
    # 计算损失函数对于第三层的权重、偏置的梯度  
    grad_W3=a2.t()@grad_z3# 权重梯度计算：δL/δW=aT@(δL/δz)(基于微积分中的链式法则)  
    grad_b3=torch.sum(grad_z3,dim=0)# 偏置梯度计算：δL/δb=I@(δL/δz),为什么这里要求和？  
    # 因为偏置项是跨样本共享的参数，同一个偏置项要加到所有的样本上。当我们更新偏置项使，要考虑其对所有样本的影响  
  
    # 计算第二层的梯度  
    grad_a2=grad_z3@model.W3.t()# z3=a2*W3+b3  
    grad_z2=grad_a2*(z2>0).float()# ReLU的导数  
    grad_W2=a1.t()@grad_z2  
    grad_b2=torch.sum(grad_z2,dim=0)  
  
    # 计算第一层的梯度  
    grad_a1=grad_z2@model.W2.t()# z2=a1*W2+b2  
    grad_z1=grad_a1*(z1>0).float()# ReLU的导数  
    grad_W1=x_flat.t()@grad_z1  
    grad_b1=torch.sum(grad_z1,dim=0)  
  
    return grad_W1,grad_b1,grad_W2,grad_b2,grad_W3,grad_b3  
  
#6.手动实现参数更新  
def manaul_update(model,grad_W1,grad_b1,grad_W2,grad_b2,grad_W3,grad_b3,lr):  
         # 更新第一层参数  
         # 参数=参数-学习率*梯度  
      with torch.no_grad():  
          lr=1e-3  
          model.W1-=lr*grad_W1  
          model.b1-=lr*grad_b1  
         # 更新第二层参数  
          model.W2-=lr*grad_W2  
          model.b2-=lr*grad_b2  
         # 更新输出层参数  
          model.W3-=lr*grad_W3  
          model.b3-=lr*grad_b3  
#7.训练循环  
model=NeuralNetwork().to(device)  
epochs=5  
  
for epoch in range(epochs):  
    model.train()  
    total_loss=0  
    for batch,(X,y)in enumerate(train_dataloader):  
        X,y=X.to(device),y.to(device)  
  
        # 手动前向传播  
        logits,cache=manual_forward(X,model)  
        # 手动计算损失  
        loss,softmax=manual_cross_entropy(logits,y)  
        loss, softmax = manual_cross_entropy(logits, y)  
        total_loss += loss.item()  
  
        # 手动反向传播  
        grads = manual_backward(X, y, logits, softmax, cache, model)  
  
        # 手动更新参数  
        manaul_update(model, *grads, lr=1e-3)  
  
        if batch % 100 == 0:  
            print(f"Epoch {epoch + 1}, Batch {batch}, Loss: {loss.item():.4f}")  
  
    avg_loss = total_loss / len(train_dataloader)  
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")  
  
    # 在每个epoch结束后测试模型  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  
        for X, y in test_dataloader:  
            X, y = X.to(device), y.to(device)  
            logits, _ = manual_forward(X, model)  
            pred = torch.argmax(logits, dim=1)  
            correct += (pred == y).sum().item()  
            total += y.size(0)  
  
    accuracy = 100 * correct / total  
    print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")  
  
print("Training completed!")  
  
#8. 保存模型  
torch.save({  
    'W1': model.W1,  
    'b1': model.b1,  
    'W2': model.W2,  
    'b2': model.b2,  
    'W3': model.W3,  
    'b3': model.b3  
}, "manual_model.pth")  
print("Saved manual model to manual_model.pth")  
  
#9. 加载模型并进行预测  
checkpoint = torch.load("manual_model.pth")  
loaded_model = NeuralNetwork().to(device)  
loaded_model.W1.data = checkpoint['W1']  
loaded_model.b1.data = checkpoint['b1']  
loaded_model.W2.data = checkpoint['W2']  
loaded_model.b2.data = checkpoint['b2']  
loaded_model.W3.data = checkpoint['W3']  
loaded_model.b3.data = checkpoint['b3']  
  
classes = [  
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",  
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"  
]  
  
#使用模型进行预测  
loaded_model.eval()  
x, y = test_data[0][0], test_data[0][1]  
with torch.no_grad():  
    x = x.to(device)  
    logits, _ = manual_forward(x.unsqueeze(0), loaded_model)  
    predicted = torch.argmax(logits, dim=1)  
    predicted_class = classes[predicted.item()]  
    actual_class = classes[y]  
    print(f'Predicted: "{predicted_class}", Actual: "{actual_class}"')


## numpy代码

import numpy as np  
import matplotlib.pyplot as plt  
  
#1.参数初始化  
def initialize_parameters(input_size,hidden_size,output_size):  
    """  
    初始化神经网络参数  
    参数：  
    input_size:输入层大小  
    hidden_size；隐藏层大小  
    output_size:输出层大小  
    返回：    包含权重和偏置的字典  
  
    """    np.random.seed(42)# 设置随机种子确保结果可以重现  
    parameters={  
        'W1':np.random.randn(hidden_size,input_size)*0.01,# 隐藏层权重  
        'b1':np.zeros((hidden_size,1)),# 隐藏层偏置项，创建一个元素都为0的列向量  
        # 偏置项是神经元的特性，不是样本的属性，一个神经元作用于m个样本的偏置项是相同的  
        'W2':np.random.randn(output_size,hidden_size)*0.01,# 输出层权重  
        'b2':np.zeros((output_size,1)),# 输出层偏置  
    }  
    return parameters  
  
#2.激活函数  
def sigmoid(z):# z就是权重*样本的结果  
    """Sigmoid激活函数"""  
    return 1/(1+np.exp(-z))  
  
def tanh(z):  
    """  
    Tanh函数  
    它是一个将任意实数输入“压缩”到 (-1, 1) 区间的S形函数。  
    数学定义：  
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))    """    return np.tanh(z)  
  
def relu(z):  
    """ReLU激活函数"""  
    return np.maximum(0,z)  
  
#3.前向传播  
def forward_progagation(X,parameters):  
    """  
    执行前向传播  
  
    参数：  
    X：输入数据  
    parameters:包含权重和偏置的字典  
  
    返回：    包含各层计算结果和缓存的元组  
    """    # 获取参数  
    W1=parameters['W1']  
    b1=parameters['b1']  
    W2=parameters['W2']  
    b2=parameters['b2']  
  
    # 第一层（隐藏层)计算  
    Z1=np.dot(W1,X)+b1# dot:矩阵相乘  
    A1=tanh(Z1)# 使用tanh激活函数，多用于隐藏层  
  
    # 第二层（输出层）计算  
    Z2=np.dot(W2,A1)+b2  
    A2=sigmoid(Z2)# 使用Sigmoid激活函数（适用于二分类）  
  
    # 缓存中间结果，适用于反向传播  
    cache={  
        'Z1':Z1,  
        'A1':A1,  
        'Z2':Z2,  
        'A2':A2  
    }  
    return A2,cache  
  
#4.计算损失  
def compute_cost(A2,Y):  
    """  
    计算交叉熵损失  
  
    参数：  
    A2：前向传播的输出  
    Y：真实标签  
  
    返回：    成本值  
    """    m=Y.shape[1]# 样本数量  
  
    # 计算交叉熵损失  
    # 对于二分类问题，交叉熵损失的公式是L(y, a) = - [y * log(a) + (1-y) * log(1-a)]  
    logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)# multiply:逐个元素相乘；log:取对数  
    cost=-np.sum(logprobs)/m# 对得到的m*1列向量中的元素求和，在/m，求出平均误差  
  
    # 确保成本是标量而不是数组  
    cost=np.squeeze(cost)  
    # squeeze的作用：删除数组中所有长度为1的维度，确保最终返回的都是一个纯粹的标量数值，而不是包裹在数组中的数值  
  
    return cost  
  
#5.反向传播  
def background_propagation(parameters,cache,X,Y):  
    """  
    执行反向传播  
  
    参数：  
    parameters: 包含权重和偏置的字典  
    cache: 前向传播的缓存  
    X:输入数据  
    Y:真实标签  
  
    返回：包含梯度的字典  
    """    m=X.shape[1]# 样本数量  
  
    # 获取参数和内存  
    W1=parameters['W1']  
    W2=parameters['W2']  
    A1=cache['A1']  
    A2=cache['A2']  
  
    # 输出层的梯度计算  
    # δL/δZ=δL/δA * δA/δz=A-Y  
    dZ2=A2-Y  
    dW2=(1/m)*np.dot(dZ2,A1.T)# δL/δW2=dZ2*(δZ2/δW2)=dZ2*(A1.T)  
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)  
  
    # 隐藏层的梯度计算  
    dZ1=np.dot(W2.T, dZ2)*(1-np.power(A1,2))# tanh的导数是1-tanh^2  
    dW1=(1/m)*np.dot(dZ1,X.T)  
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)  
  
    gradients={  
        'dW1':dW1,  
        'db1':db1,  
        'dW2':dW2,  
        'db2':db2  
    }  
  
    return gradients  
  
#6.参数更新  
def update_parameters(parameters,gradients,learning_rate):  
    """  
  
    使用梯度下降更新参数  
  
    参数：  
    parameters: 包含权重和偏置的字典  
    gradients: 包含梯度的字典  
    learning_rate: 学习率  
  
    返回：更新后的参数  
    """    # 获取参数  
    W1 = parameters['W1']  
    b1 = parameters['b1']  
    W2 = parameters['W2']  
    b2 = parameters['b2']  
  
    # 获取梯度  
    dW1 = gradients['dW1']  
    db1 = gradients['db1']  
    dW2 = gradients['dW2']  
    db2 = gradients['db2']  
  
    # 更新参数  
    W1=W1-learning_rate*dW1  
    b1=b1-learning_rate*db1  
    W2=W2-learning_rate*db2  
    b2=b2-learning_rate*b2  
  
    update_parameters={  
        'W1':W1,  
        'b1':b1,  
        'W2':W2,  
        'b2':b2  
    }  
  
    return update_parameters  
  
#7.构建完整模型  
def model(X,Y,hidden_size,learning_rate=0.01,num_iterations=10000,print_cost=False):  
    """  
    构建完整的神经网络模型  
    参数：  
    X: 输入数据  
    T: 真实标签  
    hidden_size: 隐藏层大小  
    learning_rate: 学习率  
    num_iterations: 迭代次数  
    print_cost: 是否打印成本  
  
    返回：    训练后的参数  
  
    """    np.random.seed(3)# 保证参数初始化的结果一致  
    input_size=X.shape[0]  
    output_size=Y.shape[0]  
  
  
    # 初始化参数  
    parameters=initialize_parameters(input_size,hidden_size,output_size)  
  
    costs=[]# 用于记录成本  
  
    # 训练循环  
    for i in range(num_iterations):  
        # 前向传播  
        A2, cache = forward_progagation(X, parameters)  
  
        # 计算成本  
        cost = compute_cost(A2, Y)  
  
        # 反向传播  
        gradients = background_propagation(parameters, cache, X, Y)  
  
        # 更新参数  
        parameters = update_parameters(parameters, gradients, learning_rate)  
  
        # 记录成本  
        if i % 1000 == 0:  
            costs.append(cost)  
            if print_cost:  
                print(f"迭代次数 {i}: 成本 = {cost}")  
  
        # 绘制成本曲线  
    plt.plot(cost)  
    plt.ylabel('成本')  
    plt.xlabel('迭代次数 (每千次)')  
    plt.title(f'学习率 = {learning_rate}')  
    plt.show()  
  
    return parameters  
  
    # 8. 预测函数  
  
  
def predict(parameters, X):  
    """  
    使用训练好的参数进行预测  
  
    参数:  
    parameters: 训练后的参数  
    X: 输入数据  
  
    返回:  
    预测结果 (0或1)  
    """    A2, _ = forward_progagation(X, parameters)  
    predictions = (A2 > 0.5).astype(int)  
    return predictions  
  
    # 9. 测试模型  
  
  
def test_model():  
    """测试神经网络模型"""  
    # 创建简单的数据集  
    np.random.seed(1)  
  
    # 生成两类数据点  
    class1 = np.random.randn(2, 50) + np.array([[2], [2]])  
    class2 = np.random.randn(2, 50) + np.array([[-2], [-2]])  
  
    # 合并数据  
    X = np.hstack((class1, class2))  
    Y = np.hstack((np.zeros((1, 50)), np.ones((1, 50))))  
  
    # 训练模型  
    parameters = model(X, Y, hidden_size=4, learning_rate=0.1,  
                       num_iterations=10000, print_cost=True)  
  
    # 进行预测  
    predictions = predict(parameters, X)  
  
    # 计算准确率  
    accuracy = np.mean(predictions == Y) * 100  
    print(f"训练准确率: {accuracy:.2f}%")  
  
    return parameters, X, Y  
  
# 运行测试  
if __name__ == "__main__":  
    parameters, X, Y = test_model()

## cpp神经网络
### 代码

#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<algorithm>
#include<numeric>
#include<stdexcept> // 添加stdexcept头文件用于异常处理
class NeuralNetwork{
private:
//网络结构参数
int input_size, hidden_size, output_size;
//权重和偏置
std::vector<std::vector<double>> weights_input_hidden;
std::vector<double> biases_hidden;
std::vector<std::vector<double>> weights_hidden_output;
std::vector<double> biases_output;
//前向传播中间结果
std::vector<double> hidden_layer_linear;
std::vector<double> hidden_layer_activation;
std::vector<double> output_layer_linear;
std::vector<double> output_layer_activation;
// 存储反向传播的梯度（移动到private区域）
std::vector<double> gradients_delta_output;
std::vector<double> gradients_delta_hidden;
std::vector<double> gradients_last_input;
public:
NeuralNetwork(int input_size, int hidden_size, int output_size)
: input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
srand(time(0));
}
//1.参数初始化
void initialize_parameters(){
//初始化输入层到隐藏层的权重（使用Xavier）
weights_input_hidden.resize(hidden_size, std::vector<double>(input_size));
// 修复：循环条件应该是 j < input_size 而不是 j > input_size
for(int i = 0; i < hidden_size; ++i){
for(int j = 0; j < input_size; ++j){ // 修复这里
double range = sqrt(6.0 / (input_size + hidden_size));
weights_input_hidden[i][j] = (double)rand() / RAND_MAX * 2 * range - range;
}
}
//初始化隐藏层偏置
biases_hidden.resize(hidden_size, 0.1); 
//初始化隐藏层到输出层的权重
weights_hidden_output.resize(output_size, std::vector<double>(hidden_size));
for(int i = 0; i < output_size; ++i){
for(int j = 0; j < hidden_size; ++j){
double range = sqrt(6.0 / (hidden_size + output_size));
weights_hidden_output[i][j] = (double)rand() / RAND_MAX * 2 * range - range;
}
}
//初始化输出层偏置
biases_output.resize(output_size, 0.1);
std::cout << "参数初始化完成" << std::endl;
}
//激活函数
double relu(double x){
return std::max(0.0, x);
}
double sigmoid(double x){
return 1.0 / (1.0 + exp(-x));
}
//激活函数导数
double relu_derivative(double x){
return x > 0 ? 1.0 : 0.0;
}
double sigmoid_derivative(double x){
return x * (1 - x);
}
//2.前向传播
std::vector<double> forward(const std::vector<double>& input){
//检查输入尺寸
if(input.size() != input_size){
throw std::invalid_argument("输入尺寸不匹配");
}
//隐藏层线性计算：z1 = W1*x + b1
hidden_layer_linear.resize(hidden_size);
for(int i = 0; i < hidden_size; ++i){
// 修复：这里应该是biases_hidden[i]而不是biases_output[i]
hidden_layer_linear[i] = biases_hidden[i];
for(int j = 0; j < input_size; ++j){
hidden_layer_linear[i] += input[j] * weights_input_hidden[i][j];
}
}
//隐藏层激活：a1 = relu(z1)
hidden_layer_activation.resize(hidden_size);
for(int i = 0; i < hidden_size; ++i){
hidden_layer_activation[i] = relu(hidden_layer_linear[i]);
}
//输出层线性计算：z2 = W2*a1 + b2
output_layer_linear.resize(output_size);
for(int i = 0; i < output_size; ++i){
output_layer_linear[i] = biases_output[i];
for(int j = 0; j < hidden_size; ++j){
output_layer_linear[i] += hidden_layer_activation[j] * weights_hidden_output[i][j];
}
}
//输出层激活：a2 = sigmoid(z2)
output_layer_activation.resize(output_size);
for(int i = 0; i < output_size; ++i){
output_layer_activation[i] = sigmoid(output_layer_linear[i]);
}
return output_layer_activation;
} 
//3.损失计算（二元交叉熵损失）
double compute_loss(const std::vector<double>& prediction, const std::vector<double>& target){
double loss = 0.0;
for(size_t i = 0; i < prediction.size(); ++i){
//避免log(0)的情况
double y_pred = std::max(std::min(prediction[i], 1.0 - 1e-15), 1e-15);
loss += -target[i] * log(y_pred) - (1 - target[i]) * log(1 - y_pred);
}
return loss / prediction.size();
}
//4.反向传播
void backward(const std::vector<double>& input, const std::vector<double>& target){
//计算输出层梯度
std::vector<double> delta_output(output_size);
for(int i = 0; i < output_size; ++i){
//dL/dz2 = (a2 - y) * sigmoid_derivative
delta_output[i] = (output_layer_activation[i] - target[i]) * sigmoid_derivative(output_layer_activation[i]);
} 
//计算隐藏层梯度
std::vector<double> delta_hidden(hidden_size, 0.0);
for(int i = 0; i < hidden_size; ++i){
//dL/dz1 = (W2^T * delta_output) * relu_derivative
for(int j = 0; j < output_size; ++j){
delta_hidden[i] += delta_output[j] * weights_hidden_output[j][i];
}
delta_hidden[i] *= relu_derivative(hidden_layer_activation[i]);
}
//储存梯度用于参数更新
gradients_delta_output = delta_output;
gradients_delta_hidden = delta_hidden;
gradients_last_input = input;
}  
//5.参数更新：梯度下降
void update_parameters(double learning_rate){
//更新隐藏层到输出层的权重和偏置
for(int i = 0; i < output_size; ++i){
for(int j = 0; j < hidden_size; ++j){
// 修复：这里应该是weights_hidden_output而不是weights_input_hidden
weights_hidden_output[i][j] -= learning_rate * gradients_delta_output[i] * hidden_layer_activation[j];
}
biases_output[i] -= learning_rate * gradients_delta_output[i];
}
//更新输入层到隐藏层的权重和偏置
for(int i = 0; i < hidden_size; ++i){
for(int j = 0; j < input_size; ++j){
// 修复：这里应该是weights_input_hidden而不是weights_hidden_output
weights_input_hidden[i][j] -= learning_rate * gradients_delta_hidden[i] * gradients_last_input[j];
}
biases_hidden[i] -= learning_rate * gradients_delta_hidden[i];
}
}
// 完整的训练流程
void train(const std::vector<std::vector<double>>& X,
const std::vector<std::vector<double>>& y,
int epochs, double learning_rate) {
initialize_parameters();
for(int epoch = 0; epoch < epochs; ++epoch){
double total_loss = 0.0;
for(size_t i = 0; i < X.size(); ++i){
// 前向传播
std::vector<double> prediction = forward(X[i]);
// 计算损失
double loss = compute_loss(prediction, y[i]);
total_loss += loss;
// 反向传播
backward(X[i], y[i]);
// 参数更新
update_parameters(learning_rate);
}
// 每1000轮打印一次损失
if(epoch % 1000 == 0){
std::cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size() << std::endl;
}
}
}
};
// 测试函数
int main() {
// XOR 数据集
std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};
// 创建神经网络
NeuralNetwork nn(2, 4, 1); // 2输入, 4隐藏神经元, 1输出
// 训练
nn.train(X, y, 10000, 0.5);
// 测试
std::cout << "\n测试结果:" << std::endl;
for(size_t i = 0; i < X.size(); ++i){
std::vector<double> prediction = nn.forward(X[i]);
std::cout << "输入: " << X[i][0] << ", " << X[i][1]
<< " -> 预测: " << prediction[0]
<< " (期望: " << y[i][0] << ")" << std::endl;
}
return 0;
}
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczNTUwMDU1MF19
-->
