## 分割
### 图像分割
分割任务：在原始图像中逐像素地找到需要的东西（**逐像素点**做分类）
语义分割：只区分大类，不区分大类中的具体类别
实例分割：不仅要区分大类。还要区分类别中的每一个个体。
MIOU指标：IOU（交并比），计算所有类别的平均值，一般当做分割任务评估指标
左图：计算预测值与真实值的交集；
右图：计算预测值和真实值的并集
**计算交集和并集之比，结果越接近于1，说明分割效果越好。**
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252Fs08l155b1pK6pwzL.png)
## U-net网络
网络基本结构：其实就是一个编码器加一个解码器：
下采样，编码器，利用卷积层进行特征提取；
上采样：解码器，利用插值，也是卷积层输出想要的结果（预测每一个像素点属于某一个类别的概率值）
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252F2jpuKw497sOZcyO6.png)
为什么模型一定是U形：引入拼接相同尺寸矩阵的概念来提取浅层的特征，让模型同时做到提取浅层特征和深层特征：
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252FI7R7K3rkuxX7di00.png)
## U-net++
1.非常浅层的特征和非常深层的特征拼接，跨度太大；而且要保证拼接的矩阵特征尺寸相同匹配：
所以我们融合特征采样，例如把x1.0先上采样，再和x0.0拼接。其实就是优化了特征拼接的方法。
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252FzloX5Uni7egYvW5t.png)
2.在上面融合特征采样的基础上，每一步都加上损失函数的计算和更新。
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252FhYMAgtm4VP3BtYAo.png)
优点：
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-10-05%252FI7kEJShaXr9J8D9B.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTczMzU5OTM3XX0=
-->
