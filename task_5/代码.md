### Q1:DFS
```
def depthFirstSearch(problem: Any) -> List[Any]:  
    """  
    Search the deepest nodes in the search tree first (graph search).    Returns a list of actions that reaches the goal.    """    start = problem.getStartState()  
    if problem.isGoalState(start):  
        return []  
  
    frontier = Stack()          # 存 (state, actions)    frontier.push((start, []))  
    visited = set()  
  
    while not frontier.isEmpty():  
        state, actions = frontier.pop()  
        if state in visited:  
            continue  
        visited.add(state)  
  
        if problem.isGoalState(state):  
            return actions  
  
        for nextState, action, _ in problem.getSuccessors(state):  
            if nextState not in visited:  
                frontier.push((nextState, actions + [action]))  
    return []  
    util.raiseNotDefined()
 ```
### Q2：（BFS）
```
def breadthFirstSearch(problem: Any) -> List[Any]:  
    """  
    Search the shallowest nodes in the search tree first (graph search).    Returns a list of actions that reaches the goal.    """    start = problem.getStartState()  
    if problem.isGoalState(start):  
        return []  
  
    frontier = Queue()               # 1) FIFO 队列  
    frontier.push((start, []))  
    visited = set()  
    visited.add(start)               # BFS 可以在入队时就标记，省一次 pop 判断  
  
    while not frontier.isEmpty():  
        state, actions = frontier.pop()  
  
        if problem.isGoalState(state):  
            return actions           # 2) 找到目标  
  
        for nextState, action, _ in problem.getSuccessors(state):  
            if nextState not in visited:  
                visited.add(nextState)  
                frontier.push((nextState, actions + [action]))  
  
    return []                        # 3) 无解
   ```
### Q3：变化代价函数
```
def uniformCostSearch(problem: Any) -> List[Any]:  
    """统一代价图搜索：返回累计代价最小的动作序列"""  
    start = problem.getStartState()  
    if problem.isGoalState(start):  
        return []  
  
    frontier = PriorityQueue()          # (cost, state, actions)  
    frontier.push((start, []), 0)  
    visited = set()  
  
    while not frontier.isEmpty():  
        state, actions = frontier.pop()  
        if state in visited:  
            continue  
        visited.add(state)  
  
        if problem.isGoalState(state):  
            return actions  
  
        for nextState, action, stepCost in problem.getSuccessors(state):  
            if nextState not in visited:  
                newCost = problem.getCostOfActions(actions + [action])  
                frontier.push((nextState, actions + [action]), newCost)  
  
    return []
  ```
### Q3：变化代价函数
```
def uniformCostSearch(problem: Any) -> List[Any]:  
    """统一代价图搜索：返回累计代价最小的动作序列"""  
    start = problem.getStartState()  
    if problem.isGoalState(start):  
        return []  
  
    frontier = PriorityQueue()          # (cost, state, actions)  
    frontier.push((start, []), 0)  
    visited = set()  
  
    while not frontier.isEmpty():  
        state, actions = frontier.pop()  
        if state in visited:  
            continue  
        visited.add(state)  
  
        if problem.isGoalState(state):  
            return actions  
  
        for nextState, action, stepCost in problem.getSuccessors(state):  
            if nextState not in visited:  
                newCost = problem.getCostOfActions(actions + [action])  
                frontier.push((nextState, actions + [action]), newCost)  
  
    return []
  ```


   
<!--stackedit_data:
eyJoaXN0b3J5IjpbODYzMzA0NTY0XX0=
-->