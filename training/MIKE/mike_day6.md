---

### Day 6 — 栈与队列

---

### 232. 用栈实现队列

**M - 暴力解**
```
用单个栈，每次pop时将所有元素倒入临时栈，取底部元素，再倒回来
Time: O(n) per pop, Space: O(n)
```

**I - 边界分析**
- 上界：O(n) per pop
- 下界：均摊 O(1)
- 目标：均摊 O(1)

**K - 关键词触发**
- 栈模拟队列 → 双栈（输入栈+输出栈）

**E - 优化方案**
- 双栈：输入栈负责push，输出栈负责pop/peek。输出栈空时才从输入栈倒入
- 均摊 Time: O(1), Space: O(n)

**核心代码片段**
```python
class MyQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def push(self, x):
        self.stack_in.append(x)

    def pop(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()

    def peek(self):
        val = self.pop()
        self.stack_out.append(val)
        return val

    def empty(self):
        return not self.stack_in and not self.stack_out
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class MyQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def push(self, x):
        self.stack_in.append(x)

    def pop(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()

    def peek(self):
        val = self.pop()
        self.stack_out.append(val)
        return val

    def empty(self):
        return not self.stack_in and not self.stack_out

q = MyQueue()
n = int(input())
for _ in range(n):
    line = input().split()
    if line[0] == "push":
        q.push(int(line[1]))
    elif line[0] == "pop":
        print(q.pop())
    elif line[0] == "peek":
        print(q.peek())
    elif line[0] == "empty":
        print("true" if q.empty() else "false")
```

---

### 225. 用队列实现栈

**M - 暴力解**
```
用两个队列，pop时将n-1个元素移到另一个队列，取最后一个
Time: O(n) per pop, Space: O(n)
```

**I - 边界分析**
- 上界：O(n) per pop
- 下界：O(n) per push 或 O(n) per pop（至少一个O(n)）
- 目标：O(n) per push, O(1) per pop

**K - 关键词触发**
- 队列模拟栈 → 单队列，push时旋转

**E - 优化方案**
- 单队列：push时将新元素入队后，把前面size-1个元素依次出队再入队
- Push: O(n), Pop/Top: O(1)

**核心代码片段**
```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0

s = MyStack()
n = int(input())
for _ in range(n):
    line = input().split()
    if line[0] == "push":
        s.push(int(line[1]))
    elif line[0] == "pop":
        print(s.pop())
    elif line[0] == "top":
        print(s.top())
    elif line[0] == "empty":
        print("true" if s.empty() else "false")
```

---

### 20. 有效的括号

**M - 暴力解**
```
反复替换相邻匹配括号对为空串，直到无法替换
如果最终为空则合法
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 括号匹配 → 栈
- 嵌套结构 → 栈

**E - 优化方案**
- 栈：左括号入栈，右括号与栈顶匹配则弹栈，最终栈空则合法
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def isValid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack

s = input().strip()
print("true" if isValid(s) else "false")
```

---

### 1047. 删除字符串中的所有相邻重复项

**M - 暴力解**
```
反复扫描字符串，删除相邻重复对，直到无法删除
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 相邻重复消除 → 栈（类似消消乐）

**E - 优化方案**
- 栈：遍历字符，与栈顶相同则弹出，否则入栈
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def removeDuplicates(s: str) -> str:
    stack = []
    for c in s:
        if stack and stack[-1] == c:
            stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def removeDuplicates(s):
    stack = []
    for c in s:
        if stack and stack[-1] == c:
            stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)

s = input().strip()
print(removeDuplicates(s))
```

---

### 150. 逆波兰表达式求值

**M - 暴力解**
```
直接用栈模拟即为标准解法
遇到数字入栈，遇到运算符弹出两个数计算后入栈
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 已是最优

**K - 关键词触发**
- 后缀表达式 → 栈
- 运算符优先级 → 栈

**E - 优化方案**
- 栈模拟，注意除法向零截断（Python用int(a/b)而非a//b）
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def evalRPN(tokens: list) -> int:
    stack = []
    for t in tokens:
        if t in "+-*/":
            b, a = stack.pop(), stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            else: stack.append(int(a / b))
        else:
            stack.append(int(t))
    return stack[0]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def evalRPN(tokens):
    stack = []
    for t in tokens:
        if t in "+-*/":
            b, a = stack.pop(), stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            else: stack.append(int(a / b))
        else:
            stack.append(int(t))
    return stack[0]

tokens = input().split()
print(evalRPN(tokens))
```

---

### 239. 滑动窗口最大值

**M - 暴力解**
```
每次窗口滑动时遍历窗口内所有元素找最大值
Time: O(n * k), Space: O(1)
```

**I - 边界分析**
- 上界：O(n * k)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 滑动窗口 + 最大值 → 单调递减队列
- 动态维护最值 → 单调队列/堆

**E - 优化方案**
- 单调递减队列（双端队列）：队列中存下标，维护从队首到队尾递减
- 新元素入队前弹出所有比它小的，队首超出窗口则弹出
- Time: O(n), Space: O(k)

**核心代码片段**
```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()  # 存下标
    res = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

def maxSlidingWindow(nums, k):
    dq = deque()
    res = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res

n, k = map(int, input().split())
nums = list(map(int, input().split()))
print(' '.join(map(str, maxSlidingWindow(nums, k))))
```

---

### 347. 前K个高频元素

**M - 暴力解**
```
统计频次后排序，取前K个
Time: O(n log n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n log n)
- 下界：O(n)（至少遍历一次统计频次）
- 目标：O(n log k)

**K - 关键词触发**
- 前K个/TopK → 堆（小顶堆大小K）
- 频次统计 → 哈希表

**E - 优化方案**
- 哈希计数 + 小顶堆（大小K）：遍历频次map，维护大小K的小顶堆
- Time: O(n log k), Space: O(n)
- 进阶：桶排序 O(n)

**核心代码片段**
```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    cnt = Counter(nums)
    return [x for x, _ in cnt.most_common(k)]
    # 或手动堆：
    # heap = []
    # for num, freq in cnt.items():
    #     heapq.heappush(heap, (freq, num))
    #     if len(heap) > k:
    #         heapq.heappop(heap)
    # return [x for _, x in heap]
```

**ACM模式完整代码**
```python
import sys
import heapq
from collections import Counter
input = sys.stdin.readline

def topKFrequent(nums, k):
    cnt = Counter(nums)
    heap = []
    for num, freq in cnt.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [x for _, x in heap]

nums = list(map(int, input().split()))
k = int(input())
res = topKFrequent(nums, k)
print(' '.join(map(str, res)))
```

---

### 155. 最小栈

**M - 暴力解**
```
每次getMin遍历整个栈找最小值
Time: O(n) per getMin, Space: O(n)
```

**I - 边界分析**
- 上界：O(n) per getMin
- 下界：O(1) per getMin
- 目标：O(1) 所有操作

**K - 关键词触发**
- O(1)获取最小值 + 栈 → 辅助栈同步记录最小值

**E - 优化方案**
- 辅助栈：主栈存数据，辅助栈同步存当前最小值
- 所有操作 Time: O(1), Space: O(n)

**核心代码片段**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [float('inf')]

    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [float('inf')]

    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]

ms = MinStack()
n = int(input())
for _ in range(n):
    line = input().split()
    if line[0] == "push":
        ms.push(int(line[1]))
    elif line[0] == "pop":
        ms.pop()
    elif line[0] == "top":
        print(ms.top())
    elif line[0] == "getMin":
        print(ms.getMin())
```
