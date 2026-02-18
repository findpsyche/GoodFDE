---

### Day 11 — 贪心算法（上）

---

### 455. 分发饼干

**M - 暴力解**
```
双重循环：每个孩子遍历所有饼干找最小满足的
Time: O(n*m), Space: O(1)
```

**I - 边界分析**
- 上界：O(n*m)
- 下界：O(nlogn + mlogm)（排序复杂度）
- 目标：O(nlogn + mlogm + n + m)

**K - 关键词触发**
- 分配问题+局部最优 → 贪心
- 小饼干优先满足小胃口 → 双指针+排序

**E - 优化方案**
- 排序后双指针：小饼干优先满足小胃口，满足则count++，不满足则换更大饼干
- Time: O(nlogn + mlogm), Space: O(1)

**核心代码片段**
```python
def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = count = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            count += 1
            i += 1
        j += 1
    return count
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = count = 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]:
            count += 1
            i += 1
        j += 1
    return count

g = list(map(int, input().split()))
s = list(map(int, input().split()))
print(findContentChildren(g, s))
```

---

### 376. 摆动序列

**M - 暴力解**
```
回溯枚举所有子序列，检查是否摆动
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 下界：O(n)（必须遍历一遍）
- 目标：O(n)

**K - 关键词触发**
- 摆动序列 → 贪心：统计波峰波谷个数
- 局部最优：删除单调坡度中间元素，保留峰谷

**E - 优化方案**
- 贪心：记录前一个差值preDiff，当前差值curDiff，符号变化则count++
- 特殊处理：平坡（curDiff==0）不更新preDiff，初始preDiff=0处理首元素
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def wiggleMaxLength(nums):
    if len(nums) <= 1: return len(nums)
    preDiff = 0
    curDiff = 0
    count = 1  # 默认最右边有一个峰值
    for i in range(len(nums) - 1):
        curDiff = nums[i + 1] - nums[i]
        if (preDiff <= 0 and curDiff > 0) or (preDiff >= 0 and curDiff < 0):
            count += 1
            preDiff = curDiff  # 只在摆动时更新
    return count
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def wiggleMaxLength(nums):
    if len(nums) <= 1: return len(nums)
    preDiff = 0
    curDiff = 0
    count = 1
    for i in range(len(nums) - 1):
        curDiff = nums[i + 1] - nums[i]
        if (preDiff <= 0 and curDiff > 0) or (preDiff >= 0 and curDiff < 0):
            count += 1
            preDiff = curDiff
    return count

nums = list(map(int, input().split()))
print(wiggleMaxLength(nums))
```

---

### 53. 最大子数组和

**M - 暴力解**
```
双重循环枚举所有子数组，计算和
Time: O(n^2), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 连续子数组最值 → 贪心/DP
- 贪心策略：curSum<0则重置为0（负数只会拖累后续）

**E - 优化方案**
- 贪心：维护curSum，<0则重置，每步更新maxSum
- DP视角：dp[i] = max(dp[i-1]+nums[i], nums[i])
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def maxSubArray(nums):
    curSum = 0
    maxSum = float('-inf')
    for num in nums:
        curSum += num
        maxSum = max(maxSum, curSum)
        if curSum < 0:
            curSum = 0
    return maxSum
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def maxSubArray(nums):
    curSum = 0
    maxSum = float('-inf')
    for num in nums:
        curSum += num
        maxSum = max(maxSum, curSum)
        if curSum < 0:
            curSum = 0
    return maxSum

nums = list(map(int, input().split()))
print(maxSubArray(nums))
```

---

### 122. 买卖股票的最佳时机 II

**M - 暴力解**
```
回溯枚举所有买卖组合
Time: 指数级, Space: O(n)
```

**I - 边界分析**
- 上界：指数级
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 多次买卖+最大利润 → 贪心：收集所有正利润
- 局部最优：只要明天比今天贵就今天买明天卖

**E - 优化方案**
- 贪心：遍历相邻差值，正数累加（等价于收集所有上升段）
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit

prices = list(map(int, input().split()))
print(maxProfit(prices))
```

---

### 55. 跳跃游戏

**M - 暴力解**
```
回溯/BFS枚举所有跳跃路径
Time: O(n^n), Space: O(n)
```

**I - 边界分析**
- 上界：指数级
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 能否到达 → 贪心：维护最远可达位置maxReach
- 局部最优：每步更新maxReach = max(maxReach, i + nums[i])

**E - 优化方案**
- 贪心：遍历时维护maxReach，若i > maxReach则无法到达，若maxReach >= n-1则成功
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def canJump(nums):
    maxReach = 0
    for i in range(len(nums)):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i + nums[i])
        if maxReach >= len(nums) - 1:
            return True
    return True
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def canJump(nums):
    maxReach = 0
    for i in range(len(nums)):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i + nums[i])
        if maxReach >= len(nums) - 1:
            return True
    return True

nums = list(map(int, input().split()))
print("true" if canJump(nums) else "false")
```

---

### 45. 跳跃游戏 II

**M - 暴力解**
```
BFS层序遍历，每层是一次跳跃
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 最少跳跃次数 → 贪心：每次在当前范围内选能跳最远的位置
- 双指针：curEnd标记当前跳跃边界，maxReach标记下次能到的最远

**E - 优化方案**
- 贪心+双指针：遍历到curEnd时steps++，更新curEnd=maxReach
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def jump(nums):
    if len(nums) == 1: return 0
    steps = 0
    curEnd = 0
    maxReach = 0
    for i in range(len(nums) - 1):  # 不遍历最后一个
        maxReach = max(maxReach, i + nums[i])
        if i == curEnd:
            steps += 1
            curEnd = maxReach
    return steps
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def jump(nums):
    if len(nums) == 1: return 0
    steps = 0
    curEnd = 0
    maxReach = 0
    for i in range(len(nums) - 1):
        maxReach = max(maxReach, i + nums[i])
        if i == curEnd:
            steps += 1
            curEnd = maxReach
    return steps

nums = list(map(int, input().split()))
print(jump(nums))
```

---

### 134. 加油站

**M - 暴力解**
```
枚举每个起点，模拟一圈
Time: O(n^2), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 环形数组+起点选择 → 贪心
- 总油量 >= 总消耗 → 必有解
- curSum < 0 → 起点更新为下一站（前面的站都不行）

**E - 优化方案**
- 贪心：维护totalSum和curSum，curSum<0则start=i+1，最后检查totalSum>=0
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def canCompleteCircuit(gas, cost):
    totalSum = 0
    curSum = 0
    start = 0
    for i in range(len(gas)):
        totalSum += gas[i] - cost[i]
        curSum += gas[i] - cost[i]
        if curSum < 0:
            start = i + 1
            curSum = 0
    return start if totalSum >= 0 else -1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def canCompleteCircuit(gas, cost):
    totalSum = 0
    curSum = 0
    start = 0
    for i in range(len(gas)):
        totalSum += gas[i] - cost[i]
        curSum += gas[i] - cost[i]
        if curSum < 0:
            start = i + 1
            curSum = 0
    return start if totalSum >= 0 else -1

gas = list(map(int, input().split()))
cost = list(map(int, input().split()))
print(canCompleteCircuit(gas, cost))
```

---

### 135. 分发糖果

**M - 暴力解**
```
模拟：不断调整直到满足所有约束
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 双向约束 → 两次贪心遍历
- 左到右：保证右边评分高的比左边多
- 右到左：保证左边评分高的比右边多

**E - 优化方案**
- 两次遍历贪心：
  1. 从左到右：ratings[i] > ratings[i-1] 则 candies[i] = candies[i-1] + 1
  2. 从右到左：ratings[i] > ratings[i+1] 则 candies[i] = max(candies[i], candies[i+1] + 1)
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    # 从左到右
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    # 从右到左
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)

ratings = list(map(int, input().split()))
print(candy(ratings))
```

---
