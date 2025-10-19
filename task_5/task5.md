## ask 5：具身智能

### 题目背景

-   你是一个外星人安插在地球的间谍，平日里假装自己是一种叫“LLM”的 AI 模型潜伏在人类的互联网上，偷偷地观察人类。有一天，你接触到了“图灵测试”。
-   图灵测试流程：询问者通过文本提问，另一房间中的人与机器分别回答，询问者根据回答判断真人与机器，测试隔离进行，旨在评估机器能否展现与人类无异的智能。

[![image](https://d.jotang.club/uploads/default/optimized/1X/fb78f4d4b19bdb09cfa33e38246e7ae74c08c4fc_2_690x244.png)]
-   但你发现人们总能精准无误地发现你并非人类，因为你切换话题时反应速度过快、对太多领域有超出常人的专业程度，说话过于有理有据逻辑清晰……你感叹道，哎，人类真是愚蠢🥸
-   于是你打算为自己打造一具躯体，从电脑里走出来，在现实世界和人类过过招，但在此之前，你想现在模拟器里学学怎么规划自己平日里行动的路线。

### 完成题目

-   什么是智能体（agent）？什么是智能决策，它一定要用深度学习吗？什么是具身智能？
智能体是一个**能够感知环境、做出决策、并采取行动**的系统。它可以是软件（股票交易算法、ChatGPT），也可以是硬件（如扫地机器人、自动驾驶汽车）。
智能决策就是在部分可观测，不确定，动态变化的环境中，智能体通过**推理或学习**选择**最优或次优动作**的过程。智能决策不一定要用到深度学习，当问题比较复杂，例如图像识别、自然语言理解、游戏AI是会用到深度学习。
 具身智能是智能体通过**物理身体（或模拟身体）与真实世界**持续互动，从而发展出智能的概念。例如机器人在摔倒100次之后学会走路；我的世界里先学会用手砍树，再盖房子。
-   Pacman 吃豆人 7
 -   克隆代码，切换到相应目录后，你可以在命令行中输入以下命令来玩一局 Pacman 游戏`python pacman.py`

文件类别

文件名

说明

**你需要编辑的文件**

`search.py`

所有搜索算法的实现文件

`searchAgents.py`

所有基于搜索的智能体实现文件

**你可能想查看的文件**

`pacman.py`

运行 Pacman 游戏的主文件，定义了 Pacman GameState 类型，本项目中会用到

`game.py`

Pacman 世界的逻辑，定义了支持类型如 AgentState、Agent、Direction、Grid

`util.py`

实现搜索算法时用到的实用数据结构

**可忽略的辅助文件**

`graphicsDisplay.py`

Pacman 的图形显示相关代码

`graphicsUtils.py`

支持 Pacman 图形显示的辅助代码

`textDisplay.py`

Pacman 的 ASCII 图形显示代码

`ghostAgents.py`

控制幽灵的智能体代码

`keyboardAgents.py`

键盘接口控制 Pacman 的代码

`layout.py`

读取布局文件并存储其内容的代码

`autograder.py`

项目自动评分程序

`testParser.py`

解析自动评分测试和答案文件的代码

`testClasses.py`

通用自动评分测试类

`searchTestClasses.py`

项目1特定的自动评分测试类

`test_cases/`

存放每个问题测试用例的目录

-   你将在在 `searchAgents.py` 实现数个算法，具体要求见仓库。最后请提交你的源代码、测试结果、演示视频（如有）。

![image](https://d.jotang.club/uploads/default/optimized/1X/e4afbc4ef036569771b8112acce153b1402bd74e_2_690x375.png)]
### Q1：使用深度优先搜索（DFS）寻找固定食物点

提示：DFS、BFS、UCS和A*只在边界管理策略上不同，先做好DFS，其他较容易。

实现`depthFirstSearch`函数（search.py），写一个图搜索版本，避免重复扩展已访问状态。

自动评分测试：

![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-08-24%252F2KPbXmrmaTugtSPZ.png)


### Q2：广度优先搜索（BFS）
注意：如果搜索代码写得通用，修改后应能直接用于八数码问题。
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-08-24%252FMVLMslN1CBisUOCZ.png)
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-08-24%252F298CQWvVuROtn8Mr.png)
自动评分测试。

![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-08-24%252FGAvxM6UqHJZ0Jwu8.png)

### Q3：变化代价函数
自动评分测试：
![输入图片说明](https://github.com/Lily-923/stackedit-app-data/blob/master/imgs%252F2025-08-24%252FTk4czSIZ82UrtdfY.png)

### Q4：A*搜索

示例运行：
自动评分测试：
### Q5：PPO + 神经网络
PPO（Proximal Policy Optimization）是一种强化学习算法，可以用在很多领域，包括游戏、机器人控制、自动驾驶等。

用 PPO 训练智能体，让它在游戏中学习如何在迷宫中找到食物和躲避幽灵步骤大致如下：

-   用神经网络表示策略（Policy Network），输入环境状态（比如地图、Pac-Man位置、幽灵位置等），输出动作概率分布（向左、向右、向上、向下等）。
    
-   让智能体根据策略在环境中行动，收集状态-动作-奖励数据。
    
-   用PPO算法更新神经网络，优化策略以获得更高的长期奖励。
    
-   重复训练直到智能体表现满意。
    
请你在 `searchAgents.py` 中实现你的神经网络推理，训练的代码放在另一个 `.py` 文件，在 `searchAgents.py` 中调用。最后录制运行展示视频提交即可。

### 提交说明

-   文件夹命名为“task5”，内容一并放在 GitHub 上，文件夹中应包含：
-   **文档**：你的学习笔记、实验过程的记录、验证结果截图等。
-   你的所有**代码**，及其 readme 文件。


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcxMjU3NzI3Ml19
-->
