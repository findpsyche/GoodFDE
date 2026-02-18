---

### Day 15 — 动态规划（股票 + 打家劫舍）+ 单调栈

---

### 121. 买卖股票的最佳时机

**M - 暴力解**
```
For each pair (i, j) where i < j, calculate profit = prices[j] - prices[i]
Track maximum profit
Time: O(n^2), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^2) 枚举所有买卖日对
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space
- 只能交易一次，找最大利润

**K - 关键词触发**
- 一次买卖最大利润 → 贪心：维护最低买入价，计算当前卖出利润
- 单次遍历 → 动态维护minPrice和maxProfit

**E - 优化方案**
- 遍历价格数组，维护到目前为止的最低价格minPrice
- 对每个价格计算 profit = price - minPrice，更新maxProfit
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def maxProfit(prices: list) -> int:
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    prices = list(map(int, input().split()))
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    print(max_profit)

solve()
```

---

### 123. 买卖股票的最佳时机 III

**M - 暴力解**
```
Enumerate all possible split points i
For [0, i]: compute max profit of one transaction
For [i+1, n-1]: compute max profit of one transaction
Sum and track maximum
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2) 枚举分割点
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space
- 最多两次交易，状态机DP

**K - 关键词触发**
- 最多k次交易 → 状态机DP：持有/未持有状态
- k=2 → 5个状态：未操作、第一次持有、第一次卖出、第二次持有、第二次卖出
- 状态转移 → buy1 = max(buy1, -price), sell1 = max(sell1, buy1 + price)

**E - 优化方案**
- 定义5个状态变量：
  - buy1: 第一次买入后的最大利润（负数）
  - sell1: 第一次卖出后的最大利润
  - buy2: 第二次买入后的最大利润
  - sell2: 第二次卖出后的最大利润
- 状态转移：buy1 = max(buy1, -price), sell1 = max(sell1, buy1 + price), buy2 = max(buy2, sell1 - price), sell2 = max(sell2, buy2 + price)
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def maxProfit(prices: list) -> int:
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0
    for price in prices:
        buy1 = max(buy1, -price)
        sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price)
        sell2 = max(sell2, buy2 + price)
    return sell2
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    prices = list(map(int, input().split()))
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0
    for price in prices:
        buy1 = max(buy1, -price)
        sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price)
        sell2 = max(sell2, buy2 + price)
    print(sell2)

solve()
```

---

### 198. 打家劫舍

**M - 暴力解**
```
Enumerate all subsets where no two adjacent houses are selected
Calculate sum for each valid subset, track maximum
Time: O(2^n), Space: O(n) for recursion
```

**I - 边界分析**
- 上界：O(2^n) 枚举所有子集
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space
- 不相邻元素最大和 → 经典DP

**K - 关键词触发**
- 不相邻元素最大和 → DP：dp[i] = max(偷i, 不偷i)
- 状态转移 → dp[i] = max(dp[i-1], dp[i-2] + nums[i])
- 空间优化 → 滚动变量，只需prev1和prev2

**E - 优化方案**
- 定义dp[i]为前i个房子能偷的最大金额
- 转移：dp[i] = max(dp[i-1], dp[i-2] + nums[i])
  - 不偷第i个：dp[i-1]
  - 偷第i个：dp[i-2] + nums[i]（不能偷i-1）
- 空间优化：用两个变量prev1, prev2滚动
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def rob(nums: list) -> int:
    prev1 = prev2 = 0
    for num in nums:
        prev1, prev2 = max(prev1, prev2 + num), prev1
    return prev1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    prev1 = prev2 = 0
    for num in nums:
        prev1, prev2 = max(prev1, prev2 + num), prev1
    print(prev1)

solve()
```

---

### 739. 每日温度

**M - 暴力解**
```
For each day i, scan forward to find first day j where temperatures[j] > temperatures[i]
Record j - i
Time: O(n^2), Space: O(1) excluding output
```

**I - 边界分析**
- 上界：O(n^2) 每天向后扫描
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space
- 找下一个更大元素 → 单调栈经典应用

**K - 关键词触发**
- 下一个更大元素 → 单调递减栈
- 栈存下标 → 方便计算距离
- 当前温度 > 栈顶温度 → 弹栈并计算距离

**E - 优化方案**
- 维护单调递减栈（存下标）
- 遍历温度数组：
  - 当前温度 > 栈顶温度：弹栈，计算距离 answer[栈顶] = i - 栈顶
  - 当前下标入栈
- Time: O(n)（每个元素最多入栈出栈一次）, Space: O(n)

**核心代码片段**
```python
def dailyTemperatures(temperatures: list) -> list:
    n = len(temperatures)
    answer = [0] * n
    stack = []
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev = stack.pop()
            answer[prev] = i - prev
        stack.append(i)
    return answer
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    temperatures = list(map(int, input().split()))
    n = len(temperatures)
    answer = [0] * n
    stack = []
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev = stack.pop()
            answer[prev] = i - prev
        stack.append(i)
    print(" ".join(map(str, answer)))

solve()
```

---

### 496. 下一个更大元素 I

**M - 暴力解**
```
For each element in nums1, find its position in nums2
Scan forward in nums2 to find next greater element
Time: O(m * n), Space: O(1) excluding output
```

**I - 边界分析**
- 上界：O(m * n) 对nums1每个元素在nums2中扫描
- 下界：O(m + n) 至少遍历两数组
- 目标：O(m + n) time, O(n) space
- nums1是nums2的子集 → 先处理nums2，再查询

**K - 关键词触发**
- 下一个更大元素 → 单调栈
- 查询关系 → 哈希表存储 {元素: 下一个更大元素}
- 两步：1) 单调栈处理nums2建map，2) 查询nums1

**E - 优化方案**
- 用单调递减栈处理nums2，建立哈希表 {num: nextGreater}
- 遍历nums1，查表获取结果
- Time: O(m + n), Space: O(n)

**核心代码片段**
```python
def nextGreaterElement(nums1: list, nums2: list) -> list:
    next_map = {}
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            next_map[stack.pop()] = num
        stack.append(num)
    return [next_map.get(x, -1) for x in nums1]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums1 = list(map(int, input().split()))
    nums2 = list(map(int, input().split()))
    next_map = {}
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            next_map[stack.pop()] = num
        stack.append(num)
    result = [next_map.get(x, -1) for x in nums1]
    print(" ".join(map(str, result)))

solve()
```

---

### 503. 下一个更大元素 II

**M - 暴力解**
```
For each element, scan forward (with wraparound) to find next greater
Time: O(n^2), Space: O(1) excluding output
```

**I - 边界分析**
- 上界：O(n^2) 每个元素向后扫描
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space
- 循环数组 → 遍历2n次，用取模模拟循环

**K - 关键词触发**
- 循环数组 → 遍历两遍（i % n取模）
- 下一个更大元素 → 单调递减栈
- 栈存下标 → 方便取模访问

**E - 优化方案**
- 单调递减栈，遍历2n次（i从0到2n-1）
- 用 i % n 访问实际元素
- 只在第一遍（i < n）时入栈，避免重复
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def nextGreaterElements(nums: list) -> list:
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        if i < n:
            stack.append(idx)
    return result
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        if i < n:
            stack.append(idx)
    print(" ".join(map(str, result)))

solve()
```

---

### 213. 打家劫舍 II

**M - 暴力解**
```
Enumerate all valid subsets (no adjacent, no both first and last)
Calculate sum for each, track maximum
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n) 枚举所有子集
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space
- 环形数组 → 拆分成两个线性问题

**K - 关键词触发**
- 环形数组 + 不相邻 → 拆分：[0, n-2] 和 [1, n-1]
- 不能同时选首尾 → 分别计算不选尾、不选首的情况
- 线性打家劫舍 → 复用198题的DP

**E - 优化方案**
- 环形约束：不能同时偷第0和第n-1个房子
- 拆分成两个线性问题：
  - 情况1：偷[0, n-2]范围（不考虑最后一个）
  - 情况2：偷[1, n-1]范围（不考虑第一个）
- 取两种情况的最大值
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def rob(nums: list) -> int:
    if len(nums) == 1:
        return nums[0]

    def rob_linear(arr):
        prev1 = prev2 = 0
        for num in arr:
            prev1, prev2 = max(prev1, prev2 + num), prev1
        return prev1

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    if len(nums) == 1:
        print(nums[0])
        return

    def rob_linear(arr):
        prev1 = prev2 = 0
        for num in arr:
            prev1, prev2 = max(prev1, prev2 + num), prev1
        return prev1

    result = max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
    print(result)

solve()
```

---

### 84. 柱状图中最大的矩形

**M - 暴力解**
```
For each bar i, expand left and right to find max width with height >= heights[i]
Calculate area = height * width
Time: O(n^2), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^2) 每个柱子向两边扩展
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space
- 找每个柱子能扩展的最大宽度 → 单调栈

**K - 关键词触发**
- 找左右第一个更小元素 → 单调递增栈
- 矩形面积 = 高度 × 宽度 → 高度=heights[i]，宽度=右边界-左边界-1
- 栈存下标 → 方便计算宽度

**E - 优化方案**
- 维护单调递增栈（存下标）
- 遍历柱子：
  - 当前高度 < 栈顶高度：弹栈计算面积
    - 高度 = heights[栈顶]
    - 宽度 = i - stack[-1] - 1（左边界是新栈顶，右边界是i）
  - 当前下标入栈
- 技巧：数组首尾添加哨兵0，避免边界处理
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def largestRectangleArea(heights: list) -> int:
    heights = [0] + heights + [0]
    stack = []
    max_area = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    heights = list(map(int, input().split()))
    heights = [0] + heights + [0]
    stack = []
    max_area = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    print(max_area)

solve()
```

---
