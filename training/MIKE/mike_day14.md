---

### Day 14 — 动态规划（背包 + 子序列）

---

### 1049. 最后一块石头的重量 II

**M - 暴力解**
```
Recursively try all possible combinations of smashing stones
Track minimum remaining weight
Time: O(2^n), Space: O(n) recursion depth
```

**I - 边界分析**
- 上界：O(2^n) 枚举所有组合
- 下界：O(n * sum) DP背包问题
- 目标：O(n * sum/2) time, O(sum/2) space
- 关键洞察：将石头分成两堆，使两堆重量尽可能接近，差值最小

**K - 关键词触发**
- 分成两组使差值最小 → 01背包问题，target = sum/2
- 尽可能装满容量为sum/2的背包 → dp[j] = max(dp[j], dp[j-stones[i]] + stones[i])
- 最终答案 = sum - 2 * dp[target]

**E - 优化方案**
- 转化为01背包：背包容量 = sum/2，求最多能装多少重量
- dp[j] 表示容量j最多能装的重量
- 一维滚动数组，倒序遍历
- Time: O(n * sum/2), Space: O(sum/2)

**核心代码片段**
```python
def lastStoneWeightII(stones: list) -> int:
    total = sum(stones)
    target = total // 2
    dp = [0] * (target + 1)
    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = max(dp[j], dp[j - stone] + stone)
    return total - 2 * dp[target]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    stones = list(map(int, input().split()))
    total = sum(stones)
    target = total // 2
    dp = [0] * (target + 1)
    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = max(dp[j], dp[j - stone] + stone)
    print(total - 2 * dp[target])

solve()
```

---

### 494. 目标和

**M - 暴力解**
```
Recursively try +/- for each number
Count paths that reach target
Time: O(2^n), Space: O(n) recursion depth
```

**I - 边界分析**
- 上界：O(2^n) 枚举所有+/-组合
- 下界：O(n * sum) DP背包问题
- 目标：O(n * bagSize) time, O(bagSize) space
- 关键洞察：设正数和为x，负数和为y，则 x - y = target, x + y = sum，推出 x = (sum + target) / 2

**K - 关键词触发**
- 添加+/-使和为target → 转化为01背包：选若干数使和为 (sum+target)/2
- 求方案数 → dp[j] += dp[j - nums[i]]，初始 dp[0] = 1
- 组合问题 → 外层遍历物品，内层倒序遍历容量

**E - 优化方案**
- 转化为01背包组合数问题：bagSize = (sum + target) / 2
- dp[j] 表示和为j的方案数
- 递推：dp[j] += dp[j - nums[i]]
- Time: O(n * bagSize), Space: O(bagSize)

**核心代码片段**
```python
def findTargetSumWays(nums: list, target: int) -> int:
    total = sum(nums)
    if (total + target) % 2 == 1 or abs(target) > total:
        return 0
    bag_size = (total + target) // 2
    dp = [0] * (bag_size + 1)
    dp[0] = 1
    for num in nums:
        for j in range(bag_size, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[bag_size]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    target = int(input())
    total = sum(nums)
    if (total + target) % 2 == 1 or abs(target) > total:
        print(0)
        return
    bag_size = (total + target) // 2
    dp = [0] * (bag_size + 1)
    dp[0] = 1
    for num in nums:
        for j in range(bag_size, num - 1, -1):
            dp[j] += dp[j - num]
    print(dp[bag_size])

solve()
```

---

### 518. 零钱兑换 II

**M - 暴力解**
```
Recursively try all combinations of coins
Count distinct combinations that sum to amount
Time: O(amount^n), Space: O(amount) recursion depth
```

**I - 边界分析**
- 上界：O(amount^n) 递归枚举
- 下界：O(n * amount) DP完全背包
- 目标：O(n * amount) time, O(amount) space
- 关键：完全背包组合数问题

**K - 关键词触发**
- 硬币无限使用 → 完全背包
- 求组合数（不考虑顺序）→ 外层遍历硬币，内层正序遍历金额
- dp[j] += dp[j - coin]，初始 dp[0] = 1

**E - 优化方案**
- 完全背包组合数：外层遍历硬币，内层正序遍历金额
- dp[j] 表示凑成金额j的组合数
- 递推：dp[j] += dp[j - coin]
- Time: O(n * amount), Space: O(amount)

**核心代码片段**
```python
def change(amount: int, coins: list) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    return dp[amount]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    amount = int(input())
    coins = list(map(int, input().split()))
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    print(dp[amount])

solve()
```

---

### 300. 最长递增子序列

**M - 暴力解**
```
Recursively try all subsequences, check if increasing
Track maximum length
Time: O(2^n), Space: O(n) recursion depth
```

**I - 边界分析**
- 上界：O(2^n) 枚举所有子序列
- 中界：O(n^2) DP
- 下界：O(n log n) 贪心+二分
- 目标：O(n log n) time, O(n) space

**K - 关键词触发**
- 最长递增子序列 → DP: dp[i] = max(dp[j] + 1) for j < i where nums[j] < nums[i]
- 优化到O(n log n) → 贪心+二分：维护当前最长递增序列的最小尾部元素

**E - 优化方案**
- 方案1 (DP O(n^2)): dp[i] 表示以nums[i]结尾的LIS长度
  - dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
  - Time: O(n^2), Space: O(n)
- 方案2 (贪心+二分 O(n log n)): 维护数组tails，tails[i]表示长度为i+1的递增序列的最小尾部
  - 对每个num，二分查找第一个 >= num 的位置并替换
  - Time: O(n log n), Space: O(n)

**核心代码片段**
```python
# 方案1: DP O(n^2)
def lengthOfLIS_dp(nums: list) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 方案2: 贪心+二分 O(n log n)
def lengthOfLIS(nums: list) -> int:
    from bisect import bisect_left
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

**ACM模式完整代码**
```python
import sys
from bisect import bisect_left
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    print(len(tails))

solve()
```

---

### 1143. 最长公共子序列

**M - 暴力解**
```
Recursively try all subsequences of both strings
Find longest common one
Time: O(2^(m+n)), Space: O(m+n) recursion depth
```

**I - 边界分析**
- 上界：O(2^(m+n)) 枚举所有子序列
- 下界：O(m * n) DP
- 目标：O(m * n) time, O(m * n) or O(min(m,n)) space

**K - 关键词触发**
- 最长公共子序列 → 二维DP
- text1[i] == text2[j] → dp[i][j] = dp[i-1][j-1] + 1
- text1[i] != text2[j] → dp[i][j] = max(dp[i-1][j], dp[i][j-1])

**E - 优化方案**
- 二维DP：dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的LCS长度
- 递推：
  - 相等：dp[i][j] = dp[i-1][j-1] + 1
  - 不等：dp[i][j] = max(dp[i-1][j], dp[i][j-1])
- Time: O(m * n), Space: O(m * n)，可优化为 O(min(m,n)) 滚动数组

**核心代码片段**
```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    text1 = input().strip()
    text2 = input().strip()
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    print(dp[m][n])

solve()
```

---

### 718. 最长重复子数组

**M - 暴力解**
```
For each pair of starting positions (i, j), expand while matching
Track maximum length
Time: O(m * n * min(m, n)), Space: O(1)
```

**I - 边界分析**
- 上界：O(m * n * min(m, n)) 枚举起点+扩展
- 下界：O(m * n) DP
- 目标：O(m * n) time, O(m * n) or O(min(m,n)) space

**K - 关键词触发**
- 最长重复子数组（连续） → 二维DP，注意与LCS区别（LCS是子序列，可不连续）
- nums1[i] == nums2[j] → dp[i][j] = dp[i-1][j-1] + 1
- nums1[i] != nums2[j] → dp[i][j] = 0（必须连续）

**E - 优化方案**
- 二维DP：dp[i][j] 表示以 nums1[i-1] 和 nums2[j-1] 结尾的最长公共子数组长度
- 递推：
  - 相等：dp[i][j] = dp[i-1][j-1] + 1
  - 不等：dp[i][j] = 0
- 记录全局最大值
- Time: O(m * n), Space: O(m * n)，可优化为 O(min(m,n)) 滚动数组

**核心代码片段**
```python
def findLength(nums1: list, nums2: list) -> int:
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    res = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                res = max(res, dp[i][j])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums1 = list(map(int, input().split()))
    nums2 = list(map(int, input().split()))
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    res = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                res = max(res, dp[i][j])
    print(res)

solve()
```

---

### 322. 零钱兑换

**M - 暴力解**
```
Recursively try all combinations of coins
Find minimum number of coins that sum to amount
Time: O(amount^n), Space: O(amount) recursion depth
```

**I - 边界分析**
- 上界：O(amount^n) 递归枚举
- 下界：O(n * amount) DP完全背包
- 目标：O(n * amount) time, O(amount) space
- 关键：完全背包求最小数量

**K - 关键词触发**
- 硬币无限使用 → 完全背包
- 求最少硬币数 → dp[j] = min(dp[j], dp[j - coin] + 1)
- 初始化：dp[0] = 0，其余为正无穷
- 外层遍历硬币，内层正序遍历金额

**E - 优化方案**
- 完全背包最小数量：dp[j] 表示凑成金额j的最少硬币数
- 递推：dp[j] = min(dp[j], dp[j - coin] + 1)
- 初始化：dp[0] = 0，其余为 float('inf')
- Time: O(n * amount), Space: O(amount)

**核心代码片段**
```python
def coinChange(coins: list, amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    coins = list(map(int, input().split()))
    amount = int(input())
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    print(dp[amount] if dp[amount] != float('inf') else -1)

solve()
```

---

### 72. 编辑距离

**M - 暴力解**
```
Recursively try all edit operations (insert/delete/replace)
Find minimum operations to transform word1 to word2
Time: O(3^(m+n)), Space: O(m+n) recursion depth
```

**I - 边界分析**
- 上界：O(3^(m+n)) 递归枚举所有操作
- 下界：O(m * n) DP
- 目标：O(m * n) time, O(m * n) or O(min(m,n)) space

**K - 关键词触发**
- 编辑距离 → 经典二维DP
- word1[i] == word2[j] → dp[i][j] = dp[i-1][j-1]（不需要操作）
- word1[i] != word2[j] → dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
  - dp[i-1][j] + 1: 删除word1[i]
  - dp[i][j-1] + 1: 插入word2[j]
  - dp[i-1][j-1] + 1: 替换word1[i]为word2[j]

**E - 优化方案**
- 二维DP：dp[i][j] 表示 word1[0:i] 转换为 word2[0:j] 的最少操作数
- 初始化：dp[i][0] = i（删除i个字符），dp[0][j] = j（插入j个字符）
- 递推：
  - 相等：dp[i][j] = dp[i-1][j-1]
  - 不等：dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
- Time: O(m * n), Space: O(m * n)，可优化为 O(min(m,n)) 滚动数组

**核心代码片段**
```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    word1 = input().strip()
    word2 = input().strip()
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    print(dp[m][n])

solve()
```

---
