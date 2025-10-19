## Diffusion 
### 概念讲解1
扩散模型主要包括两个步骤：前向过程（逐步加噪add noise）反向过程（去噪denoise）
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-09-30%252F3EggB5qj99h52l9l.png)
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-09-30%252FkexRntWAPIb7na9B.png)
从文本到图像的生成器：
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-09-30%252FidNwYwVf3PeiGqqf.png)
### 概念讲解2
这是Duffusion模型的大致框架：
1.Text encoder:把输入数据（图像/文本）解释成向量；
2.Generation Model:用向量生成中间产物（图片的压缩版本，可能看得懂也可能看不懂）
3.Decoder:把中间产物转化成最终产物
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FxH7GHScg1uhW6aUD.png)
下面是两个常见的扩散模型的例子，都遵循上面的基本结构：
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FuIjKrqQaGvWzjv46.png)
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FQNNF8O2eAvI9GQQq.png)
1.Text encoder对最终生成图像的影响：左图（FID越小，CLIP越大，图像越靠近右下角质量越高）
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FKpd40IKnwJdmMmc5.png)
FID：FID，全称 Fréchet Inception Distance，中文可译为“弗雷歇初始距离”。它是一种用于评估生成模型（如GAN、VAE、扩散模型）性能的指标，主要衡量生成图像的质量和多样性。
简单来说，FID计算的是“真实图像”和“生成图像”在特征空间中的分布之间的距离。**如果两个分布越接近，说明生成图像的质量和多样性越接近真实图像，FID值也就越低**。
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252F7k0t3SylKNXVgfVn.png)
CLIP，全称 Contrastive Language–Image Pre-training，它的核心思想是：通过对比学习，将图像和文本映射到同一个语义空间。
简单来说，CLIP 的目标是**判断任意一张图片和任意一段文本描述是否匹配。**
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FBXc90HcwWhqAatop.png)
3.Decoder:把中间产物转化成最终产物
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252Fb4WDg1NjwyFXLjmb.png)
2.Generation Model:用向量加上输入图像生成中间产物（图片的压缩版本，可能看得懂也可能看不懂）：具体过程就是多次denoise，图像经过一次denoise变得越来越清晰。
![输入图片说明](https://raw.githubusercontent.com/Lily-923/stackedit-app-data/master/imgs%252F2025-10-04%252FyraBaK84zNO5xQ9Q.png)


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYwOTYxMzQwOV19
-->
