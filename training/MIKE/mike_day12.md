---

### Day 12 — 贪心算法（下）+ 区间问题

---

### 860. 柠檬水找零

**M - 暴力解**
```
模拟找零过程，每次遍历所有可能的找零组合
Time: O(n * 2^k), Space: O(1)
```

**I - 边界分析**
- 上界：O(n * 2^k)（k为找零组合数）
- 下界：O(n)（必须遍历每个顾客）
- 目标：O(n)

**K - 关键词触发**
- 找零 → 贪心：优先使用大面额（保留小面额灵活性）
- 局部最优 → 全局最优

**E - 优化方案**
- 贪心策略：收到20时优先用10+5找零（保留5元），只有10不够时才用3张5元
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def lemonadeChange(bills):
    five, ten = 0, 0
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0: return False
            five -= 1
            ten += 1
        else:  # bill == 20
            if ten > 0 and five > 0:  # 优先用10+5
                ten -= 1
                five -= 1
            elif five >= 3:  # 否则用3张5
                five -= 3
            else:
                return False
    return True
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def lemonadeChange(bills):
    five, ten = 0, 0
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0: return False
            five -= 1
            ten += 1
        else:
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True

bills = list(map(int, input().split()))
print("true" if lemonadeChange(bills) else "false")
```

---

### 406. 根据身高重建队列

**M - 暴力解**
```
枚举所有排列，检查每个排列是否满足k值条件
Time: O(n! * n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n!)
- 下界：O(n log n)（排序）
- 目标：O(n^2)

**K - 关键词触发**
- 二维数组排序 → 先确定一个维度，再处理另一个维度
- 插入位置 → 按身高降序，矮的人插入不影响高的人的k值

**E - 优化方案**
- 贪心+排序：按身高h降序，h相同按k升序；然后按k值插入到结果数组
- 高个子先排好，矮个子插入时不影响已排好的高个子
- Time: O(n^2), Space: O(n)

**核心代码片段**
```python
def reconstructQueue(people):
    # 按身高降序，k升序
    people.sort(key=lambda x: (-x[0], x[1]))
    res = []
    for p in people:
        res.insert(p[1], p)  # 按k值插入
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def reconstructQueue(people):
    people.sort(key=lambda x: (-x[0], x[1]))
    res = []
    for p in people:
        res.insert(p[1], p)
    return res

n = int(input())
people = []
for _ in range(n):
    h, k = map(int, input().split())
    people.append([h, k])

for p in reconstructQueue(people):
    print(f"{p[0]} {p[1]}")
```

---

### 452. 用最少数量的箭引爆气球

**M - 暴力解**
```
枚举所有可能的射箭位置组合
Time: 指数级, Space: O(n)
```

**I - 边界分析**
- 上界：指数级
- 下界：O(n log n)（排序）
- 目标：O(n log n)

**K - 关键词触发**
- 区间重叠 → 按右端点排序，贪心选择最早结束的区间
- 最少箭数 = 最多不重叠区间数

**E - 优化方案**
- 贪心：按右端点排序，当前气球左端点>上一箭位置时需要新箭
- Time: O(n log n), Space: O(1)

**核心代码片段**
```python
def findMinArrowShots(points):
    if not points: return 0
    points.sort(key=lambda x: x[1])  # 按右端点排序
    arrows = 1
    end = points[0][1]
    for i in range(1, len(points)):
        if points[i][0] > end:  # 不重叠，需要新箭
            arrows += 1
            end = points[i][1]
    return arrows
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def findMinArrowShots(points):
    if not points: return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]
    for i in range(1, len(points)):
        if points[i][0] > end:
            arrows += 1
            end = points[i][1]
    return arrows

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append([x, y])

print(findMinArrowShots(points))
```

---

### 435. 无重叠区间

**M - 暴力解**
```
枚举所有删除区间的组合，找最少删除数
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n log n)
- 目标：O(n log n)

**K - 关键词触发**
- 无重叠区间 → 按右端点排序，保留结束早的区间（给后面留更多空间）
- 最少删除 = n - 最多不重叠区间数

**E - 优化方案**
- 贪心：按右端点排序，统计最多不重叠区间数，用总数减去即为删除数
- Time: O(n log n), Space: O(1)

**核心代码片段**
```python
def eraseOverlapIntervals(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[1])
    count = 1  # 不重叠区间数
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:  # 不重叠
            count += 1
            end = intervals[i][1]
    return len(intervals) - count
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def eraseOverlapIntervals(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[1])
    count = 1
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            count += 1
            end = intervals[i][1]
    return len(intervals) - count

n = int(input())
intervals = []
for _ in range(n):
    x, y = map(int, input().split())
    intervals.append([x, y])

print(eraseOverlapIntervals(intervals))
```

---

### 56. 合并区间

**M - 暴力解**
```
双重循环检查所有区间对是否重叠，重复合并
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n log n)（排序）
- 目标：O(n log n)

**K - 关键词触发**
- 合并区间 → 按左端点排序，一次遍历合并
- 重叠判断：当前左端点 <= 上一区间右端点

**E - 优化方案**
- 排序+一次遍历：按左端点排序，维护当前合并区间的右端点
- Time: O(n log n), Space: O(n)

**核心代码片段**
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= res[-1][1]:  # 重叠
            res[-1][1] = max(res[-1][1], intervals[i][1])
        else:
            res.append(intervals[i])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= res[-1][1]:
            res[-1][1] = max(res[-1][1], intervals[i][1])
        else:
            res.append(intervals[i])
    return res

n = int(input())
intervals = []
for _ in range(n):
    x, y = map(int, input().split())
    intervals.append([x, y])

for interval in merge(intervals):
    print(f"{interval[0]} {interval[1]}")
```

---

### 763. 划分字母区间

**M - 暴力解**
```
枚举所有可能的划分方式，检查每个片段是否满足条件
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n)（必须遍历字符串）
- 目标：O(n)

**K - 关键词触发**
- 字母只出现在一个片段 → 记录每个字母最后出现位置
- 贪心：当前位置到达所有已见字母的最远位置时可以切分

**E - 优化方案**
- 两次遍历：第一次记录每个字母最后位置，第二次贪心划分
- Time: O(n), Space: O(26) = O(1)

**核心代码片段**
```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}  # 每个字母最后位置
    res = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])  # 更新当前片段最远边界
        if i == end:  # 到达边界，切分
            res.append(end - start + 1)
            start = i + 1
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    res = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            res.append(end - start + 1)
            start = i + 1
    return res

s = input().strip()
print(' '.join(map(str, partitionLabels(s))))
```

---

### 738. 单调递增的数字

**M - 暴力解**
```
从n开始递减，检查每个数是否单调递增
Time: O(n * log n), Space: O(1)
```

**I - 边界分析**
- 上界：O(n * log n)
- 下界：O(log n)（处理数字位数）
- 目标：O(log n)

**K - 关键词触发**
- 单调递增 → 从后往前找第一个递减位置
- 贪心：该位-1，后面全填9（保证最大且单调）

**E - 优化方案**
- 贪心：转字符串，从后往前找递减位，该位-1后面全填9
- Time: O(log n), Space: O(log n)

**核心代码片段**
```python
def monotoneIncreasingDigits(n):
    s = list(str(n))
    flag = len(s)  # 标记从哪里开始填9
    for i in range(len(s) - 1, 0, -1):
        if s[i - 1] > s[i]:  # 递减
            s[i - 1] = str(int(s[i - 1]) - 1)
            flag = i
    for i in range(flag, len(s)):
        s[i] = '9'
    return int(''.join(s))
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def monotoneIncreasingDigits(n):
    s = list(str(n))
    flag = len(s)
    for i in range(len(s) - 1, 0, -1):
        if s[i - 1] > s[i]:
            s[i - 1] = str(int(s[i - 1]) - 1)
            flag = i
    for i in range(flag, len(s)):
        s[i] = '9'
    return int(''.join(s))

n = int(input())
print(monotoneIncreasingDigits(n))
```

---

### 968. 监控二叉树

**M - 暴力解**
```
枚举所有可能的摄像头放置方案，找最少数量
Time: O(2^n), Space: O(h)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n)（必须遍历所有节点）
- 目标：O(n)

**K - 关键词触发**
- 树+覆盖问题 → 后序遍历（从叶子往上）
- 贪心：从叶子父节点开始放摄像头（一个摄像头覆盖3个节点）

**E - 优化方案**
- 贪心+后序遍历：定义3种状态（0=无覆盖，1=有摄像头，2=被覆盖）
- 叶子节点返回0，父节点看到0就放摄像头返回1，看到1就返回2
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def minCameraCover(root):
    res = [0]
    def dfs(node):
        if not node: return 2  # 空节点视为被覆盖
        left = dfs(node.left)
        right = dfs(node.right)
        # 左右有一个无覆盖，当前节点放摄像头
        if left == 0 or right == 0:
            res[0] += 1
            return 1
        # 左右有一个有摄像头，当前节点被覆盖
        if left == 1 or right == 1:
            return 2
        # 左右都被覆盖，当前节点无覆盖
        return 0
    # 根节点无覆盖需要加摄像头
    if dfs(root) == 0:
        res[0] += 1
    return res[0]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(nodes):
    if not nodes or nodes[0] == 'null': return None
    root = TreeNode(int(nodes[0]))
    queue = [root]
    i = 1
    while queue and i < len(nodes):
        node = queue.pop(0)
        if i < len(nodes) and nodes[i] != 'null':
            node.left = TreeNode(int(nodes[i]))
            queue.append(node.left)
        i += 1
        if i < len(nodes) and nodes[i] != 'null':
            node.right = TreeNode(int(nodes[i]))
            queue.append(node.right)
        i += 1
    return root

def minCameraCover(root):
    res = [0]
    def dfs(node):
        if not node: return 2
        left = dfs(node.left)
        right = dfs(node.right)
        if left == 0 or right == 0:
            res[0] += 1
            return 1
        if left == 1 or right == 1:
            return 2
        return 0
    if dfs(root) == 0:
        res[0] += 1
    return res[0]

nodes = input().strip().split()
root = buildTree(nodes)
print(minCameraCover(root))
```

---
