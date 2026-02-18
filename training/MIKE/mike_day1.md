---

### Day 1 — 数组

---

### 704. 二分查找

**M - 暴力解**
```
遍历数组每个元素：
    如果当前元素等于target：
        返回下标
返回-1

时间：O(n)  空间：O(1)
```
M 伪代码 + 暴力解
**I - 边界分析**
- 上界：O(n)（线性扫描）
- 下界：O(log n)（有序数组查找的理论下界）
- 目标：O(log n)

**K - 关键词触发**
- "有序数组" → 二分查找
- "查找目标值" → 二分查找
题目目标key point 
**E - 优化方案**
#业务目标 -> #数据结构  -> #动作_算法解  #what_is_diffrient_区分海量业务需要设计的支撑以及面向那种业务流时必须要的业务和知识构建支持 
- 二分查找：每次比较中间值，缩小一半搜索范围 (面对数组 的数据结构的核心思想(优化mind))
- 时间：O(log n)  空间：O(1)

**核心代码片段**
```python
left, right = 0, len(nums) - 1
while left <= right:
# left <= right 左闭右闭的区间 进行查找
    mid = (left + right) // 2
    if nums[mid] == target: return mid
    elif nums[mid] < target: left = mid + 1
    else: right = mid - 1
```
 
**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 输入：第一行n和target，第二行n个数
n, target = map(int, input().split())
nums = list(map(int, input().split()))
print(binary_search(nums, target))
```

---

### 35. 搜索插入位置

**M - 暴力解**
```
遍历数组：
    找到第一个 >= target 的位置返回
如果没找到，返回数组长度

时间：O(n)  空间：O(1)
```

**I - 边界分析**
- 上界：O(n)
- 下界：O(log n)（有序数组）
- 目标：O(log n)

**K - 关键词触发**
- "有序数组" + "插入位置" → 二分查找（找左边界）

**E - 优化方案**
- 二分查找找第一个 >= target 的位置
- 时间：O(log n)  空间：O(1)

**核心代码片段**
```python
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    if nums[mid] >= target: right = mid - 1
    else: left = mid + 1
return left
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def search_insert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
    return left

n, target = map(int, input().split())
nums = list(map(int, input().split()))
print(search_insert(nums, target))
```

---

### 27. 移除元素

**M - 暴力解**
```
创建新数组：
    遍历原数组，不等于val的元素加入新数组
返回新数组长度

时间：O(n)  空间：O(n)
```

**I - 边界分析**
- 上界：O(n)
- 下界：O(n)（必须检查每个元素）
- 已达下界，优化空间到O(1)

**K - 关键词触发**
- "原地移除" → 快慢指针

**E - 优化方案**
- 快慢指针：慢指针记录有效位置，快指针扫描
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
slow = 0
for fast in range(len(nums)):
    if nums[fast] != val:
        nums[slow] = nums[fast]
        slow += 1
return slow
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow

n, val = map(int, input().split())
nums = list(map(int, input().split()))
k = remove_element(nums, val)
print(k)
print(' '.join(map(str, nums[:k])))
```

---

### 977. 有序数组的平方

**M - 暴力解**
```
每个元素平方后排序

时间：O(n log n)  空间：O(n)
```

**I - 边界分析**
- 上界：O(n log n)
- 下界：O(n)（必须输出n个元素）
- 目标：O(n)

**K - 关键词触发**
- "有序数组" + "平方" → 对撞双指针（两端绝对值最大）

**E - 优化方案**
- 对撞双指针：两端绝对值大的先放入结果末尾
- 时间：O(n)  空间：O(n)

**核心代码片段**
```python
left, right = 0, len(nums) - 1
result = [0] * len(nums)
pos = len(nums) - 1
while left <= right:
    if abs(nums[left]) >= abs(nums[right]):
        result[pos] = nums[left] ** 2
        left += 1
    else:
        result[pos] = nums[right] ** 2
        right -= 1
    pos -= 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def sorted_squares(nums):
    n = len(nums)
    result = [0] * n
    left, right, pos = 0, n - 1, n - 1
    while left <= right:
        if abs(nums[left]) >= abs(nums[right]):
            result[pos] = nums[left] ** 2
            left += 1
        else:
            result[pos] = nums[right] ** 2
            right -= 1
        pos -= 1
    return result

n = int(input())
nums = list(map(int, input().split()))
print(' '.join(map(str, sorted_squares(nums))))
```

---

### 209. 长度最小的子数组

**M - 暴力解**
```
枚举所有子数组：
    对每个起点i，遍历终点j：
        累加和 >= target 时记录长度

时间：O(n²)  空间：O(1)
```

**I - 边界分析**
- 上界：O(n²)
- 下界：O(n)（至少遍历一次）
- 目标：O(n)

**K - 关键词触发**
- "连续子数组" + "最小长度" + "和>=target" → 滑动窗口

**E - 优化方案**
- 滑动窗口：右指针扩张累加，满足条件时左指针收缩
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
left = 0
cur_sum = 0
min_len = float('inf')
for right in range(len(nums)):
    cur_sum += nums[right]
    while cur_sum >= target:
        min_len = min(min_len, right - left + 1)
        cur_sum -= nums[left]
        left += 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def min_sub_array_len(target, nums):
    left = 0
    cur_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        cur_sum += nums[right]
        while cur_sum >= target:
            min_len = min(min_len, right - left + 1)
            cur_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0

n, target = map(int, input().split())
nums = list(map(int, input().split()))
print(min_sub_array_len(target, nums))
```

---

### 59. 螺旋矩阵 II

**M - 暴力解**
```
模拟螺旋路径填数：
    设定四条边界，按右→下→左→上顺序填入1到n²

时间：O(n²)  空间：O(n²)
```

**I - 边界分析**
- 上界 = 下界 = O(n²)（必须填n²个格子）
- 已是最优

**K - 关键词触发**
- "螺旋" + "矩阵" → 模拟，维护四条边界

**E - 优化方案**
- 统一左闭右开的边界处理，避免重复填写角落
- 时间：O(n²)  空间：O(n²)

**核心代码片段**
```python
top, bottom, left, right = 0, n-1, 0, n-1
num = 1
while top <= bottom and left <= right:
    for j in range(left, right+1): matrix[top][j] = num; num += 1
    top += 1
    for i in range(top, bottom+1): matrix[i][right] = num; num += 1
    right -= 1
    for j in range(right, left-1, -1): matrix[bottom][j] = num; num += 1
    bottom -= 1
    for i in range(bottom, top-1, -1): matrix[i][left] = num; num += 1
    left += 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num; num += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num; num += 1
        right -= 1
        for j in range(right, left - 1, -1):
            matrix[bottom][j] = num; num += 1
        bottom -= 1
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = num; num += 1
        left += 1
    return matrix

n = int(input())
matrix = generate_matrix(n)
for row in matrix:
    print(' '.join(map(str, row)))
```

---

### 76. 最小覆盖子串

**M - 暴力解**
```
枚举所有子串：
    对每个子串检查是否包含t的所有字符
    记录最短的合法子串

时间：O(n²·m)  空间：O(m)
```

**I - 边界分析**
- 上界：O(n²·m)
- 下界：O(n+m)（至少遍历s和t各一次）
- 目标：O(n)

**K - 关键词触发**
- "子串" + "包含所有字符" + "最短" → 滑动窗口 + 哈希计数

**E - 优化方案**
- 滑动窗口：右指针扩张直到覆盖t，左指针收缩求最短
- 用need计数器和formed变量判断是否覆盖
- 时间：O(n+m)  空间：O(m)

**核心代码片段**
```python
from collections import Counter
need = Counter(t)
missing = len(t)
left = start = 0
min_len = float('inf')
for right, c in enumerate(s):
    if need[c] > 0:
        missing -= 1
    need[c] -= 1
    while missing == 0:
        if right - left + 1 < min_len:
            min_len = right - left + 1
            start = left
        need[s[left]] += 1
        if need[s[left]] > 0:
            missing += 1
        left += 1
```

**ACM模式完整代码**
```python
import sys
from collections import Counter
input = sys.stdin.readline

def min_window(s, t):
    if not t or not s:
        return ""
    need = Counter(t)
    missing = len(t)
    left = start = 0
    min_len = float('inf')
    for right, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        while missing == 0:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                start = left
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return "" if min_len == float('inf') else s[start:start + min_len]

s = input().strip()
t = input().strip()
print(min_window(s, t))
```

---

### 54. 螺旋矩阵

**M - 暴力解**
```
模拟螺旋遍历：维护四条边界，按右→下→左→上读取

时间：O(m·n)  空间：O(1)（不算输出）
```

**I - 边界分析**
- 上界 = 下界 = O(m·n)
- 已是最优

**K - 关键词触发**
- "螺旋顺序" → 模拟 + 四边界

**E - 优化方案**
- 同暴力解，注意边界收缩后的判断防止重复遍历
- 时间：O(m·n)  空间：O(1)

**核心代码片段**
```python
top, bottom, left, right = 0, m-1, 0, n-1
while top <= bottom and left <= right:
    for j in range(left, right+1): res.append(matrix[top][j])
    top += 1
    for i in range(top, bottom+1): res.append(matrix[i][right])
    right -= 1
    if top <= bottom:
        for j in range(right, left-1, -1): res.append(matrix[bottom][j])
        bottom -= 1
    if left <= right:
        for i in range(bottom, top-1, -1): res.append(matrix[i][left])
        left += 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def spiral_order(matrix):
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    res = []
    top, bottom, left, right = 0, m - 1, 0, n - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            res.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1
    return res

m, n = map(int, input().split())
matrix = []
for _ in range(m):
    matrix.append(list(map(int, input().split())))
print(' '.join(map(str, spiral_order(matrix))))
```
