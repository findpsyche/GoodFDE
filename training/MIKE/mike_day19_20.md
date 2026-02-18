---

### Day 19 — 查漏补缺（补充新题）

---

### 153. 寻找旋转排序数组中的最小值

**M - 暴力解**

```
scan entire array, track minimum
Time: O(n), Space: O(1)
```

**I - 边界分析**

- 上界：O(n) 线性扫描
- 下界：O(log n)（有序性质可利用）
- 目标：O(log n)

**K - 关键词触发**

- "旋转排序数组" → 二分搜索变体
- "最小值" → 二分比较 mid 与 right 缩小范围

**E - 优化方案**

二分搜索：比较 nums[mid] 与 nums[right]。
- 若 nums[mid] > nums[right]，最小值在右半段，left = mid + 1
- 否则最小值在左半段（含mid），right = mid

Time: O(log n), Space: O(1)

**核心代码片段**

```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]

nums = list(map(int, input().split()))
print(findMin(nums))
```

---

### 162. 寻找峰值

**M - 暴力解**

```
scan array, find index where nums[i] > nums[i-1] and nums[i] > nums[i+1]
treat boundaries as -infinity
Time: O(n), Space: O(1)
```

**I - 边界分析**

- 上界：O(n) 线性扫描
- 下界：O(log n)（题目要求）
- 目标：O(log n)

**K - 关键词触发**

- "峰值" + "O(log n)" → 二分搜索
- 关键性质：nums[-1] = nums[n] = -∞，所以沿上升方向一定能找到峰值

**E - 优化方案**

二分搜索：比较 nums[mid] 与 nums[mid+1]。
- 若 nums[mid] < nums[mid+1]，峰值在右侧，left = mid + 1
- 否则峰值在左侧（含mid），right = mid

Time: O(log n), Space: O(1)

**核心代码片段**

```python
def findPeakElement(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def findPeakElement(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left

nums = list(map(int, input().split()))
print(findPeakElement(nums))
```

---

### 138. 随机链表的复制

**M - 暴力解**

```
for each node, create copy
for each node, scan entire list to find random target position
Time: O(n^2), Space: O(n)
```

**I - 边界分析**

- 上界：O(n^2) 暴力定位 random
- 下界：O(n)（每个节点至少访问一次）
- 目标：O(n)

**K - 关键词触发**

- "复制" + "random指针" → 哈希map映射旧→新节点
- 备选：原地复制拆分法（O(1)额外空间）

**E - 优化方案**

方法一（哈希map）：第一遍建立 old→new 映射，第二遍连接 next 和 random。
方法二（原地复制）：在每个节点后插入副本 → 连接random → 拆分。

Time: O(n), Space: O(n)（方法一）/ O(1)（方法二，不计输出）

**核心代码片段**

```python
# 方法一：哈希map
def copyRandomList(head):
    if not head:
        return None
    old_to_new = {}
    cur = head
    while cur:
        old_to_new[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:
        old_to_new[cur].next = old_to_new.get(cur.next)
        old_to_new[cur].random = old_to_new.get(cur.random)
        cur = cur.next
    return old_to_new[head]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def build_list(vals, random_indices):
    """vals: list of int, random_indices: list of int (-1 means None)"""
    if not vals:
        return None
    nodes = [Node(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, ri in enumerate(random_indices):
        if ri != -1:
            nodes[i].random = nodes[ri]
    return nodes[0]

def copyRandomList(head):
    if not head:
        return None
    old_to_new = {}
    cur = head
    while cur:
        old_to_new[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:
        old_to_new[cur].next = old_to_new.get(cur.next)
        old_to_new[cur].random = old_to_new.get(cur.random)
        cur = cur.next
    return old_to_new[head]

def print_list(head, n):
    cur = head
    # 收集所有节点用于定位random索引
    nodes = []
    tmp = head
    while tmp:
        nodes.append(tmp)
        tmp = tmp.next
    results = []
    for node in nodes:
        ri = -1
        if node.random:
            for j, nd in enumerate(nodes):
                if nd is node.random:
                    ri = j
                    break
        results.append(f"[{node.val},{ri}]")
    print(",".join(results))

# 输入格式：第一行n，第二行n个val，第三行n个random_index（-1表示None）
n = int(input())
if n == 0:
    print("")
else:
    vals = list(map(int, input().split()))
    random_indices = list(map(int, input().split()))
    head = build_list(vals, random_indices)
    new_head = copyRandomList(head)
    print_list(new_head, n)
```

---

### 61. 旋转链表

**M - 暴力解**

```
repeat k times: remove tail, insert at head
Time: O(n*k), Space: O(1)
```

**I - 边界分析**

- 上界：O(n*k) 逐次旋转
- 下界：O(n)（至少遍历一次获取长度）
- 目标：O(n)
- 注意：k 可能远大于 n，需取模

**K - 关键词触发**

- "旋转链表" → 首尾相连成环 + 在正确位置断开
- k % n 消除多余旋转

**E - 优化方案**

1. 遍历获取长度 n，同时到达尾节点
2. 尾节点连接头节点形成环
3. 从头走 n - k%n 步到达新尾节点
4. 新尾节点的 next 是新头，断开环

Time: O(n), Space: O(1)

**核心代码片段**

```python
def rotateRight(head, k):
    if not head or not head.next or k == 0:
        return head
    # 求长度，tail指向尾节点
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    k %= length
    if k == 0:
        return head
    # 成环
    tail.next = head
    # 走 length - k 步找到新尾
    steps = length - k
    new_tail = head
    for _ in range(steps - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(vals):
    if not vals:
        return None
    head = ListNode(vals[0])
    cur = head
    for v in vals[1:]:
        cur.next = ListNode(v)
        cur = cur.next
    return head

def print_list(head):
    res = []
    while head:
        res.append(str(head.val))
        head = head.next
    print(" ".join(res))

def rotateRight(head, k):
    if not head or not head.next or k == 0:
        return head
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    k %= length
    if k == 0:
        return head
    tail.next = head
    steps = length - k
    new_tail = head
    for _ in range(steps - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head

# 输入：第一行为链表元素（空格分隔），第二行为k
vals = list(map(int, input().split()))
k = int(input())
head = build_list(vals)
new_head = rotateRight(head, k)
print_list(new_head)
```

---

### 115. 不同的子序列

**M - 暴力解**

```
enumerate all subsequences of s, count matches with t
Time: O(2^n), Space: O(n)
```

**I - 边界分析**

- 上界：O(2^n) 枚举子序列
- 下界：O(m*n)（两个字符串匹配的经典下界）
- 目标：O(m*n)，其中 m=len(s), n=len(t)

**K - 关键词触发**

- "子序列" + "计数" → 二维DP
- dp[i][j] = s[:i] 的子序列中匹配 t[:j] 的个数

**E - 优化方案**

定义 dp[i][j]：s 的前 i 个字符中，t 的前 j 个字符作为子序列出现的次数。

转移：
- 若 s[i-1] == t[j-1]：dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
  - 用 s[i-1] 匹配 t[j-1] + 不用 s[i-1]
- 否则：dp[i][j] = dp[i-1][j]

初始化：dp[i][0] = 1（空串是任何串的子序列）

Time: O(m*n), Space: O(m*n)，可优化至 O(n)

**核心代码片段**

```python
def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]

s = input().strip()
t = input().strip()
print(numDistinct(s, t))
```

---

### 516. 最长回文子序列

**M - 暴力解**

```
enumerate all subsequences, check if palindrome, track max length
Time: O(2^n * n), Space: O(n)
```

**I - 边界分析**

- 上界：O(2^n * n) 枚举
- 下界：O(n^2)（区间DP经典下界）
- 目标：O(n^2)

**K - 关键词触发**

- "回文" + "子序列" + "最长" → 区间DP
- dp[i][j] = s[i..j] 中最长回文子序列长度

**E - 优化方案**

区间DP：dp[i][j] 表示 s[i..j] 的最长回文子序列长度。

转移：
- 若 s[i] == s[j]：dp[i][j] = dp[i+1][j-1] + 2
- 否则：dp[i][j] = max(dp[i+1][j], dp[i][j-1])

初始化：dp[i][i] = 1

遍历顺序：i 从大到小，j 从小到大（保证子问题先算）

Time: O(n^2), Space: O(n^2)

**核心代码片段**

```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]

s = input().strip()
print(longestPalindromeSubseq(s))
```

---

### 130. 被围绕的区域

**M - 暴力解**

```
for each 'O', BFS/DFS check if it can reach boundary
if cannot reach → flip to 'X'
Time: O(m*n * m*n) worst case, Space: O(m*n)
```

**I - 边界分析**

- 上界：O((mn)^2) 每个O都做一次BFS
- 下界：O(m*n)（每个格子至少看一次）
- 目标：O(m*n)

**K - 关键词触发**

- "围绕" + "边界" → 逆向思维：从边界出发标记不被围绕的O
- "区域填充" → DFS/BFS

**E - 优化方案**

逆向DFS/BFS：
1. 从四条边界上的所有 'O' 出发，DFS/BFS 标记所有与边界相连的 'O' 为 '#'
2. 遍历整个矩阵：'O' → 'X'（被围绕），'#' → 'O'（恢复）

Time: O(m*n), Space: O(m*n)（递归栈/队列）

**核心代码片段**

```python
from collections import deque

def solve(board):
    if not board:
        return
    m, n = len(board), len(board[0])

    def bfs(r, c):
        queue = deque([(r, c)])
        board[r][c] = '#'
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and board[nx][ny] == 'O':
                    board[nx][ny] = '#'
                    queue.append((nx, ny))

    # 从边界出发
    for i in range(m):
        for j in range(n):
            if (i == 0 or i == m - 1 or j == 0 or j == n - 1) and board[i][j] == 'O':
                bfs(i, j)
    # 还原
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'
```

**ACM模式完整代码**

```python
import sys
from collections import deque
input = sys.stdin.readline

def solve(board):
    if not board:
        return
    m, n = len(board), len(board[0])

    def bfs(r, c):
        queue = deque([(r, c)])
        board[r][c] = '#'
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and board[nx][ny] == 'O':
                    board[nx][ny] = '#'
                    queue.append((nx, ny))

    for i in range(m):
        for j in range(n):
            if (i == 0 or i == m - 1 or j == 0 or j == n - 1) and board[i][j] == 'O':
                bfs(i, j)
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'

# 输入：第一行 m n，接下来m行每行n个字符（空格分隔）
m, n = map(int, input().split())
board = []
for _ in range(m):
    board.append(input().split())
solve(board)
for row in board:
    print(" ".join(row))
```

---

### 417. 太平洋大西洋水流问题

**M - 暴力解**

```
for each cell, BFS/DFS check if it can reach Pacific AND Atlantic
Time: O(m*n * m*n), Space: O(m*n)
```

**I - 边界分析**

- 上界：O((mn)^2) 每个格子做两次BFS
- 下界：O(m*n)
- 目标：O(m*n)

**K - 关键词触发**

- "水流" + "两个海洋" → 逆向BFS/DFS：从海洋边界逆流而上
- "取交集" → 两次BFS结果求交

**E - 优化方案**

逆向BFS：
1. 从太平洋边界（上边+左边）出发，BFS标记所有能到达太平洋的格子（逆流：只走 >= 当前高度的邻居）
2. 从大西洋边界（下边+右边）出发，同样BFS
3. 两个集合的交集即为答案

Time: O(m*n), Space: O(m*n)

**核心代码片段**

```python
from collections import deque

def pacificAtlantic(heights):
    if not heights:
        return []
    m, n = len(heights), len(heights[0])

    def bfs(starts):
        reachable = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n
                        and (nr, nc) not in reachable
                        and heights[nr][nc] >= heights[r][c]):
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
        return reachable

    pacific = [(i, 0) for i in range(m)] + [(0, j) for j in range(1, n)]
    atlantic = [(i, n - 1) for i in range(m)] + [(m - 1, j) for j in range(n - 1)]

    pac_reach = bfs(pacific)
    atl_reach = bfs(atlantic)
    return sorted(pac_reach & atl_reach)
```

**ACM模式完整代码**

```python
import sys
from collections import deque
input = sys.stdin.readline

def pacificAtlantic(heights):
    if not heights:
        return []
    m, n = len(heights), len(heights[0])

    def bfs(starts):
        reachable = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n
                        and (nr, nc) not in reachable
                        and heights[nr][nc] >= heights[r][c]):
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
        return reachable

    pacific = [(i, 0) for i in range(m)] + [(0, j) for j in range(1, n)]
    atlantic = [(i, n - 1) for i in range(m)] + [(m - 1, j) for j in range(n - 1)]

    pac_reach = bfs(pacific)
    atl_reach = bfs(atlantic)
    return sorted(pac_reach & atl_reach)

# 输入：第一行 m n，接下来m行每行n个整数
m, n = map(int, input().split())
heights = []
for _ in range(m):
    heights.append(list(map(int, input().split())))
result = pacificAtlantic(heights)
for r, c in result:
    print(r, c)
```

---

### 380. O(1)时间插入、删除和获取随机元素

**M - 暴力解**

```
use a list for storage
insert: append O(1)
remove: scan + delete O(n)
getRandom: random.choice O(1)
```

**I - 边界分析**

- 上界：O(n) 删除需要线性扫描
- 下界：O(1) 三个操作均摊
- 目标：三个操作均为 O(1)

**K - 关键词触发**

- "O(1)插入删除" → 哈希map
- "O(1)随机获取" → 数组 + random
- 组合：数组 + 哈希map（val→index映射）

**E - 优化方案**

数组 + 哈希map：
- insert：append到数组末尾，记录 val→index
- remove：将待删元素与数组末尾交换，pop末尾，更新map
- getRandom：random.randint 取数组随机下标

三个操作均为 O(1)。Space: O(n)

**核心代码片段**

```python
import random

class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.val_to_idx = {}

    def insert(self, val):
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val):
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last = self.nums[-1]
        self.nums[idx] = last
        self.val_to_idx[last] = idx
        self.nums.pop()
        del self.val_to_idx[val]
        return True

    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]
```

**ACM模式完整代码**

```python
import sys
import random
input = sys.stdin.readline

class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.val_to_idx = {}

    def insert(self, val):
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val):
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last = self.nums[-1]
        self.nums[idx] = last
        self.val_to_idx[last] = idx
        self.nums.pop()
        del self.val_to_idx[val]
        return True

    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]

# 输入：第一行操作数n
# 接下来n行，每行格式：操作名 [参数]
# insert val / remove val / getRandom
n = int(input())
rs = RandomizedSet()
for _ in range(n):
    parts = input().split()
    op = parts[0]
    if op == "insert":
        print(rs.insert(int(parts[1])))
    elif op == "remove":
        print(rs.remove(int(parts[1])))
    elif op == "getRandom":
        print(rs.getRandom())
```
