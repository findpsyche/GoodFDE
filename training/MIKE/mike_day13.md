---

### Day 13 — 动态规划（基础）

---

### 509. 斐波那契数

**M - 暴力解**
```
递归：fib(n) = fib(n-1) + fib(n-2)
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)（递归树指数级）
- 下界：O(n)（必须计算n个状态）
- 目标：O(n)

**K - 关键词触发**
- 递归+重复子问题 → DP/记忆化
- 状态转移明确 → dp[i] = dp[i-1] + dp[i-2]

**E - 优化方案**
- DP：dp[i]表示第i个斐波那契数
- 空间优化：只需保存前两个状态，O(1)空间
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def fib(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def fib(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

n = int(input())
print(fib(n))
```

---

### 70. 爬楼梯

**M - 暴力解**
```
递归：climb(n) = climb(n-1) + climb(n-2)
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 爬楼梯/跳台阶 → 本质是斐波那契数列
- 到达第i阶 = 从i-1阶跳1步 + 从i-2阶跳2步

**E - 优化方案**
- DP：dp[i] = dp[i-1] + dp[i-2]
- 空间优化：滚动变量
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

n = int(input())
print(climbStairs(n))
```

---

### 746. 使用最小花费爬楼梯

**M - 暴力解**
```
递归：minCost(i) = min(minCost(i-1)+cost[i-1], minCost(i-2)+cost[i-2])
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 最小花费 → DP求最优解
- 可从i-1或i-2到达 → 取min

**E - 优化方案**
- DP：dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
- dp[i]表示到达第i阶的最小花费
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def minCostClimbingStairs(cost):
    n = len(cost)
    a, b = 0, 0  # dp[0], dp[1]
    for i in range(2, n + 1):
        a, b = b, min(b + cost[i - 1], a + cost[i - 2])
    return b
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def minCostClimbingStairs(cost):
    n = len(cost)
    a, b = 0, 0
    for i in range(2, n + 1):
        a, b = b, min(b + cost[i - 1], a + cost[i - 2])
    return b

cost = list(map(int, input().split()))
print(minCostClimbingStairs(cost))
```

---

### 62. 不同路径

**M - 暴力解**
```
递归：paths(i,j) = paths(i-1,j) + paths(i,j-1)
Time: O(2^(m+n)), Space: O(m+n)
```

**I - 边界分析**
- 上界：O(2^(m+n))
- 下界：O(m*n)（必须填满整个表）
- 目标：O(m*n)

**K - 关键词触发**
- 网格路径 → 二维DP
- 只能向右/向下 → dp[i][j] = dp[i-1][j] + dp[i][j-1]

**E - 优化方案**
- 二维DP：dp[i][j]表示到达(i,j)的路径数
- 空间优化：滚动数组O(n)
- Time: O(m*n), Space: O(n)

**核心代码片段**
```python
def uniquePaths(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[n - 1]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def uniquePaths(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[n - 1]

m, n = map(int, input().split())
print(uniquePaths(m, n))
```

---

### 63. 不同路径 II

**M - 暴力解**
```
递归+障碍物判断
Time: O(2^(m+n)), Space: O(m+n)
```

**I - 边界分析**
- 上界：O(2^(m+n))
- 下界：O(m*n)
- 目标：O(m*n)

**K - 关键词触发**
- 网格路径+障碍物 → 二维DP，障碍物处dp=0
- 初始化注意：第一行/列遇到障碍物后续全为0

**E - 优化方案**
- 二维DP：dp[i][j] = 0 if obstacle else dp[i-1][j] + dp[i][j-1]
- 空间优化：原地修改或滚动数组
- Time: O(m*n), Space: O(n)

**核心代码片段**
```python
def uniquePathsWithObstacles(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    if obstacleGrid[0][0] == 1: return 0
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    return dp[n - 1]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def uniquePathsWithObstacles(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    if obstacleGrid[0][0] == 1: return 0
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    return dp[n - 1]

m, n = map(int, input().split())
grid = []
for _ in range(m):
    grid.append(list(map(int, input().split())))
print(uniquePathsWithObstacles(grid))
```

---

### 343. 整数拆分

**M - 暴力解**
```
递归枚举所有拆分方式，求最大乘积
Time: O(n^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^n)
- 下界：O(n^2)（枚举拆分点）
- 目标：O(n^2)

**K - 关键词触发**
- 整数拆分+最大乘积 → DP
- 拆分i：枚举j∈[1,i-1]，选择j*(i-j)或j*dp[i-j]

**E - 优化方案**
- DP：dp[i] = max(j * (i-j), j * dp[i-j]) for j in [1, i-1]
- dp[i]表示拆分i的最大乘积
- Time: O(n^2), Space: O(n)

**核心代码片段**
```python
def integerBreak(n):
    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
    return dp[n]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def integerBreak(n):
    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
    return dp[n]

n = int(input())
print(integerBreak(n))
```

---

### 96. 不同的二叉搜索树

**M - 暴力解**
```
递归枚举每个节点作为根，左右子树方案数相乘
Time: 卡特兰数级别, Space: O(n)
```

**I - 边界分析**
- 上界：卡特兰数C(n) = C(2n,n)/(n+1)
- 下界：O(n^2)（填表）
- 目标：O(n^2)

**K - 关键词触发**
- BST数量 → 卡特兰数，DP
- 选j为根 → 左子树i=j-1个节点，右子树i=i-j个节点
- dp[i] = Σ(dp[j-1] * dp[i-j]) for j in [1,i]

**E - 优化方案**
- DP：dp[i]表示i个节点的BST数量
- Time: O(n^2), Space: O(n)

**核心代码片段**
```python
def numTrees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    return dp[n]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def numTrees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    return dp[n]

n = int(input())
print(numTrees(n))
```

---

### 416. 分割等和子集

**M - 暴力解**
```
回溯枚举所有子集，检查和是否为sum/2
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n * sum)（01背包）
- 目标：O(n * sum)

**K - 关键词触发**
- 子集和问题 → 转化为01背包
- 能否凑出target=sum/2 → dp[j]表示能否凑出和j
- 01背包：每个元素选或不选

**E - 优化方案**
- 01背包DP：dp[j] = dp[j] or dp[j - nums[i]]
- 倒序遍历j避免重复使用
- Time: O(n * target), Space: O(target)

**核心代码片段**
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def canPartition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]

nums = list(map(int, input().split()))
print("true" if canPartition(nums) else "false")
```

---
