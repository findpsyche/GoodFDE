### Day 16 — 图论 + Phase 3 测试

---

### 1. 200. 岛屿数量

**M - 暴力解**
遍历每个格子，遇到 `'1'` 就启动 DFS/BFS 把整座岛标记为已访问，计数器 +1。时间 O(m*n)，空间 O(m*n) 最坏递归栈。

**I - 边界分析**
- m, n ∈ [1, 300]，最多 90000 格，O(m*n) 可行
- 注意 grid 中是字符 `'1'` / `'0'`，不是整数

**K - 关键词触发**
- 网格连通分量 → DFS/BFS flood fill
- "数量" → 计数连通分量

**E - 优化方案**
- 原地修改 grid 把 `'1'` 改 `'0'` 省去 visited 数组
- 也可用并查集，但 DFS 更简洁

**核心代码片段**
```python
def dfs(i, j):
    if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
        return
    grid[i][j] = '0'
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        dfs(i+di, j+dj)
```

**ACM模式完整代码**
```python
import sys
from collections import deque

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    m = int(input_data[idx]); idx += 1
    n = int(input_data[idx]); idx += 1
    grid = []
    for i in range(m):
        row = list(input_data[idx]); idx += 1
        grid.append(row)

    def dfs(i, j):
        stack = [(i, j)]
        grid[i][j] = '0'
        while stack:
            x, y = stack.pop()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1':
                    grid[nx][ny] = '0'
                    stack.append((nx, ny))

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    print(count)

solve()
```

---

### 2. 695. 岛屿的最大面积

**M - 暴力解**
与 200 题类似，DFS/BFS 遍历每座岛，记录每座岛的面积，取最大值。O(m*n)。

**I - 边界分析**
- m, n ∈ [1, 50]，最多 2500 格
- grid 中是整数 1/0
- 全 0 时返回 0

**K - 关键词触发**
- 网格 + 面积 → DFS/BFS flood fill + 计数
- "最大" → 维护全局 max

**E - 优化方案**
- 原地标记避免 visited
- DFS 返回面积值，代码更简洁

**核心代码片段**
```python
def dfs(i, j):
    if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0:
        return 0
    grid[i][j] = 0
    return 1 + sum(dfs(i+di, j+dj) for di, dj in [(1,0),(-1,0),(0,1),(0,-1)])
```

**ACM模式完整代码**
```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    m = int(input_data[idx]); idx += 1
    n = int(input_data[idx]); idx += 1
    grid = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(int(input_data[idx])); idx += 1
        grid.append(row)

    def dfs(i, j):
        stack = [(i, j)]
        grid[i][j] = 0
        area = 0
        while stack:
            x, y = stack.pop()
            area += 1
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 0
                    stack.append((nx, ny))
        return area

    ans = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                ans = max(ans, dfs(i, j))
    print(ans)

solve()
```

---

### 3. 547. 省份数量

**M - 暴力解**
n 个城市的邻接矩阵，求连通分量数。DFS 遍历每个未访问城市，启动搜索标记所有可达城市。O(n²)。

**I - 边界分析**
- n ∈ [1, 200]，O(n²) = 40000 完全可行
- isConnected[i][i] = 1（自连接）
- 对称矩阵

**K - 关键词触发**
- 连通分量数 → 并查集 / DFS
- 邻接矩阵 → 直接遍历行

**E - 优化方案**
- 并查集：路径压缩 + 按秩合并，初始 n 个分量，每次 union 减 1
- DFS 更直观，并查集更通用

**核心代码片段**
```python
# 并查集
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py
```

**ACM模式完整代码**
```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    n = int(input_data[idx]); idx += 1
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(input_data[idx])); idx += 1
        matrix.append(row)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] == 1:
                union(i, j)

    provinces = len(set(find(i) for i in range(n)))
    print(provinces)

solve()
```

---

### 4. 207. 课程表

**M - 暴力解**
n 门课，prerequisites 给出先修关系。判断能否完成所有课程 = 判断有向图是否有环。BFS 拓扑排序（Kahn 算法）：入度为 0 的入队，逐步削减，最终看是否所有节点都被处理。O(V+E)。

**I - 边界分析**
- numCourses ∈ [1, 2000]，prerequisites 长度 ∈ [0, 5000]
- 可能有重复边
- 无先修条件时直接返回 True

**K - 关键词触发**
- 先修关系 + 能否完成 → 有向图环检测 → 拓扑排序
- "课程表" → 经典 Kahn 算法

**E - 优化方案**
- BFS Kahn 比 DFS 染色法更直观
- 用 defaultdict 建邻接表

**核心代码片段**
```python
from collections import deque, defaultdict
indegree = [0] * numCourses
graph = defaultdict(list)
for a, b in prerequisites:
    graph[b].append(a)
    indegree[a] += 1
queue = deque(i for i in range(numCourses) if indegree[i] == 0)
count = 0
while queue:
    node = queue.popleft()
    count += 1
    for nei in graph[node]:
        indegree[nei] -= 1
        if indegree[nei] == 0:
            queue.append(nei)
return count == numCourses
```

**ACM模式完整代码**
```python
import sys
from collections import deque, defaultdict

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    numCourses = int(input_data[idx]); idx += 1
    numPrereqs = int(input_data[idx]); idx += 1

    indegree = [0] * numCourses
    graph = defaultdict(list)
    for _ in range(numPrereqs):
        a = int(input_data[idx]); idx += 1
        b = int(input_data[idx]); idx += 1
        graph[b].append(a)
        indegree[a] += 1

    queue = deque(i for i in range(numCourses) if indegree[i] == 0)
    count = 0
    while queue:
        node = queue.popleft()
        count += 1
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)

    print("true" if count == numCourses else "false")

solve()
```

---

### 5. 210. 课程表 II

**M - 暴力解**
与 207 完全相同的 Kahn 算法，只是额外记录出队顺序作为拓扑序。若存在环则返回空数组。O(V+E)。

**I - 边界分析**
- 同 207 题
- 有环时输出空列表
- 拓扑序不唯一，任意合法序即可

**K - 关键词触发**
- "课程表 + 顺序" → 拓扑排序 + 记录序列
- Kahn 算法天然产出拓扑序

**E - 优化方案**
- 在 207 基础上加一个 order 列表即可
- DFS 后序反转也可以，但 BFS 更直观

**核心代码片段**
```python
order = []
while queue:
    node = queue.popleft()
    order.append(node)
    for nei in graph[node]:
        indegree[nei] -= 1
        if indegree[nei] == 0:
            queue.append(nei)
return order if len(order) == numCourses else []
```

**ACM模式完整代码**
```python
import sys
from collections import deque, defaultdict

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    numCourses = int(input_data[idx]); idx += 1
    numPrereqs = int(input_data[idx]); idx += 1

    indegree = [0] * numCourses
    graph = defaultdict(list)
    for _ in range(numPrereqs):
        a = int(input_data[idx]); idx += 1
        b = int(input_data[idx]); idx += 1
        graph[b].append(a)
        indegree[a] += 1

    queue = deque(i for i in range(numCourses) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)

    if len(order) == numCourses:
        print(" ".join(map(str, order)))
    else:
        print("")

solve()
```

---

### 6. 399. 除法求值

**M - 暴力解**
给定 a/b = k，求任意 x/y。建带权有向图：a→b 权 k，b→a 权 1/k。查询 x/y 就是从 x 到 y 的路径上权值之积。BFS/DFS 搜索路径。O(Q*(V+E))。

**I - 边界分析**
- equations 长度 ∈ [1, 20]，queries 长度 ∈ [1, 20]
- 变量不存在时返回 -1.0
- x == y 且 x 存在时返回 1.0
- 值均为正数，不会出现 0 除

**K - 关键词触发**
- 除法 + 传递关系 → 带权图
- 路径乘积 → BFS/DFS 搜索
- 也可用带权并查集

**E - 优化方案**
- 带权并查集：find 时路径压缩同时累乘权值，union 时计算权值比
- BFS 更直观，数据量小无需优化

**核心代码片段**
```python
# BFS 查询 src/dst
def query(src, dst):
    if src not in graph or dst not in graph:
        return -1.0
    if src == dst:
        return 1.0
    visited = {src}
    queue = deque([(src, 1.0)])
    while queue:
        node, prod = queue.popleft()
        for nei, w in graph[node]:
            if nei == dst:
                return prod * w
            if nei not in visited:
                visited.add(nei)
                queue.append((nei, prod * w))
    return -1.0
```

**ACM模式完整代码**
```python
import sys
from collections import defaultdict, deque

def solve():
    input_data = sys.stdin.read().split()
    idx = 0

    n_eq = int(input_data[idx]); idx += 1
    n_q = int(input_data[idx]); idx += 1

    graph = defaultdict(list)
    for _ in range(n_eq):
        a = input_data[idx]; idx += 1
        b = input_data[idx]; idx += 1
        val = float(input_data[idx]); idx += 1
        graph[a].append((b, val))
        graph[b].append((a, 1.0 / val))

    def query(src, dst):
        if src not in graph or dst not in graph:
            return -1.0
        if src == dst:
            return 1.0
        visited = {src}
        q = deque([(src, 1.0)])
        while q:
            node, prod = q.popleft()
            for nei, w in graph[node]:
                if nei == dst:
                    return prod * w
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, prod * w))
        return -1.0

    results = []
    for _ in range(n_q):
        x = input_data[idx]; idx += 1
        y = input_data[idx]; idx += 1
        results.append(query(x, y))

    print(" ".join(f"{r:.5f}" for r in results))

solve()
```

---

### 7. 253. 会议室 II

**M - 暴力解**
给定会议时间区间，求最少需要多少间会议室 = 求最大重叠区间数。按开始时间排序，用最小堆维护各会议室的结束时间。O(n log n)。

**I - 边界分析**
- n ∈ [1, 10⁴]
- 区间可能完全重叠
- [1,5] 和 [5,10] 不算重叠（可复用）

**K - 关键词触发**
- 区间重叠 + 最少资源 → 贪心 + 最小堆
- "会议室" → 经典扫描线 / 堆

**E - 优化方案**
- 方法一：最小堆，堆顶是最早结束的会议室
- 方法二：差分/扫描线，start +1, end -1，求前缀和最大值
- 两者都是 O(n log n)

**核心代码片段**
```python
import heapq
intervals.sort()
heap = []  # 各会议室结束时间
for start, end in intervals:
    if heap and heap[0] <= start:
        heapq.heappop(heap)  # 复用会议室
    heapq.heappush(heap, end)
return len(heap)
```

**ACM模式完整代码**
```python
import sys
import heapq

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    n = int(input_data[idx]); idx += 1
    intervals = []
    for _ in range(n):
        s = int(input_data[idx]); idx += 1
        e = int(input_data[idx]); idx += 1
        intervals.append((s, e))

    intervals.sort()
    heap = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    print(len(heap))

solve()
```

---

### 8. 139. 单词拆分

**M - 暴力解**
回溯：从位置 0 开始，尝试每个字典单词匹配前缀，匹配成功则递归剩余部分。最坏 O(2^n)。

**I - 边界分析**
- s 长度 ∈ [1, 300]，wordDict 长度 ∈ [1, 1000]，单词长度 ∈ [1, 20]
- 需要 DP 或记忆化，纯回溯会超时
- 字典中可能有重复单词

**K - 关键词触发**
- "能否拆分" → 布尔 DP
- 前缀匹配 → dp[i] 表示 s[:i] 能否被拆分
- 字典 → 转 set 加速查找

**E - 优化方案**
- DP：dp[i] = any(dp[j] and s[j:i] in wordSet for j in range(i))
- 优化内层循环：只枚举 j ∈ [max(0, i-maxLen), i]，其中 maxLen 是字典最长单词长度
- O(n * maxLen) ≈ O(300 * 20) = 6000

**核心代码片段**
```python
word_set = set(wordDict)
max_len = max(len(w) for w in wordDict)
dp = [False] * (n + 1)
dp[0] = True
for i in range(1, n + 1):
    for j in range(max(0, i - max_len), i):
        if dp[j] and s[j:i] in word_set:
            dp[i] = True
            break
return dp[n]
```

**ACM模式完整代码**
```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    s = input_data[idx]; idx += 1
    k = int(input_data[idx]); idx += 1
    word_dict = []
    for _ in range(k):
        word_dict.append(input_data[idx]); idx += 1

    word_set = set(word_dict)
    max_len = max(len(w) for w in word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    print("true" if dp[n] else "false")

solve()
```

---

### 9. 152. 乘积最大子数组

**M - 暴力解**
枚举所有子数组 O(n²)，计算乘积取最大。

**I - 边界分析**
- n ∈ [1, 2*10⁴]，元素 ∈ [-10, 10]
- 含 0 会截断乘积
- 负数 * 负数 = 正数，所以必须同时跟踪最小值
- 至少包含一个元素

**K - 关键词触发**
- "子数组 + 乘积 + 最大" → 类似 Kadane 但要维护 min/max
- 负数翻转 → 同时维护 curMax 和 curMin

**E - 优化方案**
- 对每个位置：newMax = max(num, curMax*num, curMin*num)，newMin = min(num, curMax*num, curMin*num)
- O(n) 时间 O(1) 空间

**核心代码片段**
```python
cur_max = cur_min = ans = nums[0]
for num in nums[1:]:
    candidates = (num, cur_max * num, cur_min * num)
    cur_max = max(candidates)
    cur_min = min(candidates)
    ans = max(ans, cur_max)
return ans
```

**ACM模式完整代码**
```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    n = int(input_data[idx]); idx += 1
    nums = []
    for _ in range(n):
        nums.append(int(input_data[idx])); idx += 1

    cur_max = cur_min = ans = nums[0]
    for num in nums[1:]:
        candidates = (num, cur_max * num, cur_min * num)
        cur_max = max(candidates)
        cur_min = min(candidates)
        ans = max(ans, cur_max)

    print(ans)

solve()
```

---

### 10. 85. 最大矩形

**M - 暴力解**
枚举所有矩形左上角和右下角 O(m²n²)，检查是否全 1，O(mn)。总 O(m³n³) 不可行。

**I - 边界分析**
- m, n ∈ [0, 200]，O(m*n²) 可行
- matrix 中是字符 `'0'` / `'1'`
- 空矩阵返回 0

**K - 关键词触发**
- "最大矩形 + 01矩阵" → 逐行累积高度 + 84题柱状图最大矩形
- 84题 → 单调栈 O(n)

**E - 优化方案**
- 逐行构建 heights 数组：heights[j] = (heights[j]+1) if matrix[i][j]=='1' else 0
- 对每行的 heights 调用 84 题的单调栈算法
- 总时间 O(m*n)

**核心代码片段**
```python
def largestRectangleInHistogram(heights):
    stack = [-1]
    max_area = 0
    for i, h in enumerate(heights):
        while stack[-1] != -1 and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    while stack[-1] != -1:
        height = heights[stack.pop()]
        width = len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    return max_area
```

**ACM模式完整代码**
```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    m = int(input_data[idx]); idx += 1
    n = int(input_data[idx]); idx += 1

    if m == 0 or n == 0:
        print(0)
        return

    matrix = []
    for i in range(m):
        row = list(input_data[idx]); idx += 1
        matrix.append(row)

    def largest_rect_histogram(heights):
        stack = [-1]
        max_area = 0
        for i, h in enumerate(heights):
            while stack[-1] != -1 and heights[stack[-1]] >= h:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        while stack[-1] != -1:
            height = heights[stack.pop()]
            width = len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        return max_area

    heights = [0] * n
    ans = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        ans = max(ans, largest_rect_histogram(heights))

    print(ans)

solve()
```

---

### 11. 785. 判断二分图

**M - 暴力解**
尝试用两种颜色给图染色，BFS/DFS 遍历。若相邻节点同色则不是二分图。O(V+E)。

**I - 边界分析**
- n ∈ [1, 100]，边数 ∈ [0, n*(n-1)/2]
- 图可能不连通，需要对每个连通分量都检查
- 无自环，无重边

**K - 关键词触发**
- "二分图" → 二染色 BFS/DFS
- 奇数环 → 不是二分图

**E - 优化方案**
- BFS 染色最直观
- 也可用并查集：对每个节点，其所有邻居应在同一集合（与自己不同集合）

**核心代码片段**
```python
color = [0] * n  # 0=未染色, 1/-1=两色
for i in range(n):
    if color[i] != 0:
        continue
    queue = deque([i])
    color[i] = 1
    while queue:
        node = queue.popleft()
        for nei in graph[node]:
            if color[nei] == 0:
                color[nei] = -color[node]
                queue.append(nei)
            elif color[nei] == color[node]:
                return False
return True
```

**ACM模式完整代码**
```python
import sys
from collections import deque

def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    n = int(input_data[idx]); idx += 1
    e = int(input_data[idx]); idx += 1

    graph = [[] for _ in range(n)]
    for _ in range(e):
        u = int(input_data[idx]); idx += 1
        v = int(input_data[idx]); idx += 1
        graph[u].append(v)
        graph[v].append(u)

    color = [0] * n

    def bfs(start):
        queue = deque([start])
        color[start] = 1
        while queue:
            node = queue.popleft()
            for nei in graph[node]:
                if color[nei] == 0:
                    color[nei] = -color[node]
                    queue.append(nei)
                elif color[nei] == color[node]:
                    return False
        return True

    for i in range(n):
        if color[i] == 0:
            if not bfs(i):
                print("false")
                return

    print("true")

solve()
```
