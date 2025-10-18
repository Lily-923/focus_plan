## Task 6: Vision-Language-Action

### 题目背景

-   现实世界中的道路交通状况充满了各种意想不到的复杂情形。除了模型未曾见过的复杂场景外，还可能遇到需优先遵循交警指挥、在特定情况下临时违反交通规则，或根据临时路牌进行路径调整等情况。近年来迅速发展的大语言模型（LLM）和视觉-语言模型（VLM）被认为能够有效应对这些挑战，这也正是视觉-语言-行动（VLA）研究兴起的重要动因之一。
    
-   阅读[论文 4]
    
-   这篇论文用 LLM，利用纯文字实现了感知 - 决策 - 控制的全过程，不具备很强的落地能力，但是利用 LLM 较好的探索。
    

论文将模拟器中的场景用规定格式的文字描述，随后送入LLM进行推理，最后用文字输出合适的驾驶策略。

[![image](https://d.jotang.club/uploads/default/optimized/1X/f28712bbe679ed3f76057378c057700eb2ddeca0_2_690x431.jpeg)](https://d.jotang.club/uploads/default/original/1X/f28712bbe679ed3f76057378c057700eb2ddeca0.jpeg "image")

image1820×1138 172 KB

### [](https://d.jotang.club/t/topic/976#h-32)回答问题

-   为什么本篇工作要在模拟器里进行，而不是在真实世界数据集？说说你的理解。
-   你觉得本篇论文的感知阶段要为 LLM 描述哪些信息？
-   如果你看了 Task 4，你觉得在决策阶段，应该让 LLM 形式化推理更好，还是不加格式、长度约束更好？
-   如果你看了 Task 5，你觉得在控制阶段，LLM 的输出足以控制汽车安全驾驶吗？你觉得可以怎么改进这部分？

### [](https://d.jotang.club/t/topic/976#h-33)简单实践

-   [这里 1](https://github.com/JoTang/JotangRecrument/tree/main/ML/task_6/data) 包含了数个交通场景。请你用 **VLM**（可调用API或部署到本地/服务器，不要使用网页服务），**不加训练**地用视觉-语言-行动实现一个感知-规划-控制闭环，要求使用形式化推理，得出驾驶策略（至少包含刹车、油门、方向盘的操作）。

### [](https://d.jotang.club/t/topic/976#h-34)提交说明

-   文件夹命名为“task3”，内容一并放在 GitHub 上，文件夹中应包含：
-   **文档**：你的学习笔记、实验过程的记录、验证结果截图等。
-   你的所有**代码**，及其 readme 文件。


> Written with [StackEdit中文版](https://stackedit.cn/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ1MzA0NTE3N119
-->