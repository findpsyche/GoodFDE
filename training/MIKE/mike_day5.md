---

### Day 5 — 双指针 + Phase 1 测试

---

### 283. 移动零

> 将数组中所有0移到末尾，保持非零元素相对顺序

**M - 暴力解**

```
For each 0 found, shift all elements after it left by 1, place 0 at end
Repeat until no more 0s in the non-zero portion
Time: O(n^2), Space: O(1)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n^2) 暴力移位 |
| 下界 | O(n) 每个元素至少看一次 |
| 目标 | O(n) time, O(1) space, in-place |

**K - 关键词触发**

- "移动/交换/原地" → 双指针
- "保持相对顺序" → 快慢指针（慢指针记录写入位置）

**E - 优化方案**

快慢双指针：slow 指向下一个非零元素应放的位置，fast 遍历数组。遇到非零就交换 slow/fast，slow++。

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def moveZeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    nums = list(map(int, input().split()))
    slow = 0
    for fast in range(n):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    print(' '.join(map(str, nums)))

solve()
```

---

### 844. 比较含退格的字符串

> 比较两个含退格符(#)的字符串处理后是否相等

**M - 暴力解**

```
Use a stack for each string:
  For each char, if '#' then pop, else push
Compare two resulting stacks
Time: O(n + m), Space: O(n + m)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n+m) time, O(n+m) space（栈模拟） |
| 下界 | O(n+m) time（每个字符至少看一次） |
| 目标 | O(n+m) time, O(1) space |

**K - 关键词触发**

- "退格/回退" → 从后往前处理
- "比较两个序列" → 双指针从尾部同步遍历

**E - 优化方案**

从后往前双指针：各维护一个 skip 计数器，遇到 `#` 则 skip++，遇到普通字符且 skip>0 则跳过并 skip--。两个指针同步比较当前有效字符。

- Time: O(n + m)
- Space: O(1)

**核心代码片段**

```python
def backspaceCompare(s, t):
    i, j = len(s) - 1, len(t) - 1
    skip_s = skip_t = 0
    while i >= 0 or j >= 0:
        while i >= 0:
            if s[i] == '#':
                skip_s += 1; i -= 1
            elif skip_s > 0:
                skip_s -= 1; i -= 1
            else:
                break
        while j >= 0:
            if t[j] == '#':
                skip_t += 1; j -= 1
            elif skip_t > 0:
                skip_t -= 1; j -= 1
            else:
                break
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            return False
        i -= 1; j -= 1
    return True
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    s = input().strip()
    t = input().strip()
    i, j = len(s) - 1, len(t) - 1
    skip_s = skip_t = 0
    while i >= 0 or j >= 0:
        while i >= 0:
            if s[i] == '#':
                skip_s += 1; i -= 1
            elif skip_s > 0:
                skip_s -= 1; i -= 1
            else:
                break
        while j >= 0:
            if t[j] == '#':
                skip_t += 1; j -= 1
            elif skip_t > 0:
                skip_t -= 1; j -= 1
            else:
                break
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                print("false")
                return
        elif i >= 0 or j >= 0:
            print("false")
            return
        i -= 1; j -= 1
    print("true")

solve()
```

---

### 11. 盛最多水的容器

> 选两条线使围成的水面积最大

**M - 暴力解**

```
Try all pairs (i, j) where i < j
area = min(height[i], height[j]) * (j - i)
Track maximum area
Time: O(n^2), Space: O(1)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n^2) 枚举所有对 |
| 下界 | O(n) 每条线至少看一次 |
| 目标 | O(n) time, O(1) space |

**K - 关键词触发**

- "两端/最大面积" → 左右对撞双指针
- "宽度递减时高度必须递增才可能更优" → 贪心移动较短边

**E - 优化方案**

左右双指针：从两端开始，每次移动较短的那一边（因为移动较长边不可能增大面积）。

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def maxArea(height):
    l, r = 0, len(height) - 1
    ans = 0
    while l < r:
        ans = max(ans, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return ans
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    height = list(map(int, input().split()))
    l, r = 0, n - 1
    ans = 0
    while l < r:
        ans = max(ans, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    print(ans)

solve()
```

---

### 42. 接雨水

> 计算柱状图中能接住的雨水总量

**M - 暴力解**

```
For each bar i, find max height to its left and right
water at i = min(left_max, right_max) - height[i]
Sum all positive contributions
Time: O(n^2), Space: O(1)
```

**I - 边界分析**

| 项目  | 值                     |
| --- | --------------------- |
| 上界  | O(n^2) 每个位置扫左右最大值     |
| 下界  | O(n) 每个柱子至少看一次        |
| 目标  | O(n) time, O(1) space |

**K - 关键词触发**

- "接雨水/凹槽" → 经典双指针 or 单调栈
- "左右最大值决定水位" → 对撞双指针维护 left_max / right_max

**E - 优化方案**

对撞双指针：维护 left_max 和 right_max。若 left_max < right_max，则左侧是瓶颈，处理左指针位置的水量并右移；反之处理右指针。

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def trap(height):
    l, r = 0, len(height) - 1
    left_max = right_max = 0
    ans = 0
    while l < r:
        left_max = max(left_max, height[l])
        right_max = max(right_max, height[r])
        if left_max < right_max:
            ans += left_max - height[l]
            l += 1
        else:
            ans += right_max - height[r]
            r -= 1
    return ans
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    height = list(map(int, input().split()))
    if n < 3:
        print(0)
        return
    l, r = 0, n - 1
    left_max = right_max = 0
    ans = 0
    while l < r:
        left_max = max(left_max, height[l])
        right_max = max(right_max, height[r])
        if left_max < right_max:
            ans += left_max - height[l]
            l += 1
        else:
            ans += right_max - height[r]
            r -= 1
    print(ans)

solve()
```

---

### 234. 回文链表

> 判断链表是否为回文结构（快慢指针找中点+反转后半段）

**M - 暴力解**

```
Copy all values to an array
Check if array is a palindrome (two pointers from both ends)
Time: O(n), Space: O(n)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n) time, O(n) space（复制到数组） |
| 下界 | O(n) time（必须遍历所有节点） |
| 目标 | O(n) time, O(1) space |

**K - 关键词触发**

- "链表+回文" → 快慢指针找中点 + 反转后半段
- "O(1)空间" → 原地反转链表

**E - 优化方案**

1. 快慢指针找中点（slow 到中间，fast 到末尾）
2. 反转后半段链表
3. 双指针从头和中间同时比较
4. （可选）恢复链表结构

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def isPalindrome(head):
    # 快慢指针找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # 反转后半段
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
    # 比较前半段和反转后的后半段
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt

def build_list(arr):
    dummy = ListNode(0)
    cur = dummy
    for v in arr:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def solve():
    vals = list(map(int, input().split()))
    head = build_list(vals)
    # 快慢指针找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # 反转后半段
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
    # 比较
    left, right = head, prev
    while right:
        if left.val != right.val:
            print("false")
            return
        left = left.next
        right = right.next
    print("true")

solve()
```

---

### 88. 合并两个有序数组

> 将两个有序数组合并（原地存入nums1，逆向双指针）

**M - 暴力解**

```
Copy nums2 into the tail of nums1
Sort the entire nums1
Time: O((m+n)log(m+n)), Space: O(1) (in-place sort) or O(m+n) (sort internal)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O((m+n)log(m+n)) 排序 |
| 下界 | O(m+n) 每个元素至少处理一次 |
| 目标 | O(m+n) time, O(1) space, in-place |

**K - 关键词触发**

- "两个有序+合并" → 归并
- "原地/nums1有足够空间" → 逆向双指针（从后往前填充，避免覆盖）

**E - 优化方案**

逆向双指针：p1 指向 nums1 有效末尾(m-1)，p2 指向 nums2 末尾(n-1)，p 指向 nums1 总末尾(m+n-1)。每次取较大值放到 p 位置。

- Time: O(m + n)
- Space: O(1)

**核心代码片段**

```python
def merge(nums1, m, nums2, n):
    p1, p2, p = m - 1, n - 1, m + n - 1
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    m, n = map(int, input().split())
    nums1 = list(map(int, input().split()))  # 长度 m+n，后n个为0
    nums2 = list(map(int, input().split()))  # 长度 n
    p1, p2, p = m - 1, n - 1, m + n - 1
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    print(' '.join(map(str, nums1)))

solve()
```

---

### 34. 在排序数组中查找元素的第一个和最后一个位置

> 两次二分查找

**M - 暴力解**

```
Linear scan from left to find first occurrence
Linear scan from right to find last occurrence
Time: O(n), Space: O(1)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n) 线性扫描 |
| 下界 | O(log n) 已排序数组 |
| 目标 | O(log n) time, O(1) space |

**K - 关键词触发**

- "排序数组+查找" → 二分查找
- "第一个和最后一个" → 两次二分（左边界+右边界）

**E - 优化方案**

两次二分：
- 第一次找左边界（第一个 >= target 的位置，验证是否等于 target）
- 第二次找右边界（最后一个 <= target 的位置，或第一个 > target 的位置减1）

- Time: O(log n)
- Space: O(1)

**核心代码片段**

```python
def searchRange(nums, target):
    def bisect_left(nums, target):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def bisect_right(nums, target):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    left = bisect_left(nums, target)
    right = bisect_right(nums, target) - 1
    if left <= right and left < len(nums) and nums[left] == target:
        return [left, right]
    return [-1, -1]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    n, target = map(int, input().split())
    nums = list(map(int, input().split()))

    def bisect_left(nums, target):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def bisect_right(nums, target):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    left = bisect_left(nums, target)
    right = bisect_right(nums, target) - 1
    if left <= right and left < n and nums[left] == target:
        print(left, right)
    else:
        print(-1, -1)

solve()
```

---

### 92. 反转链表 II

> 反转链表中从位置left到right的部分

**M - 暴力解**

```
Copy all values to array
Reverse the subarray from index left-1 to right-1
Rebuild the linked list
Time: O(n), Space: O(n)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n) time, O(n) space（复制到数组） |
| 下界 | O(n) time（至少遍历到right位置） |
| 目标 | O(n) time, O(1) space, 一趟扫描 |

**K - 关键词触发**

- "反转链表的一部分" → 头插法（在left前面不断插入后续节点）
- "left/right位置" → 哨兵节点(dummy head)简化边界

**E - 优化方案**

一趟扫描头插法：
1. 用 dummy 节点处理 left=1 的边界
2. 找到 left 前一个节点 pre
3. 用头插法：不断把 cur.next 摘出来插到 pre 后面，重复 right-left 次

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def reverseBetween(head, left, right):
    dummy = ListNode(0, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt
    return dummy.next
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt

def build_list(arr):
    dummy = ListNode(0)
    cur = dummy
    for v in arr:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def print_list(head):
    res = []
    while head:
        res.append(str(head.val))
        head = head.next
    print(' '.join(res))

def solve():
    vals = list(map(int, input().split()))
    left, right = map(int, input().split())
    head = build_list(vals)

    dummy = ListNode(0, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt

    print_list(dummy.next)

solve()
```

---

### 560. 和为K的子数组

> 前缀和+哈希map

**M - 暴力解**

```
Try all subarrays (i, j), compute sum, check if equals k
Time: O(n^2) with prefix sum, O(n^3) without
Space: O(1) or O(n)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n^2) 枚举所有子数组 |
| 下界 | O(n) 每个元素至少看一次 |
| 目标 | O(n) time, O(n) space |

**K - 关键词触发**

- "子数组和" → 前缀和
- "和为K/计数" → 前缀和 + 哈希表（two-sum 变体）
- prefix[j] - prefix[i] = k → 查找 prefix[j] - k 是否在哈希表中

**E - 优化方案**

前缀和 + 哈希表：维护前缀和的出现次数。对于当前前缀和 cur_sum，查找 cur_sum - k 在哈希表中出现了多少次，即为以当前位置结尾的满足条件的子数组数量。

- Time: O(n)
- Space: O(n)

**核心代码片段**

```python
from collections import defaultdict

def subarraySum(nums, k):
    count = 0
    cur_sum = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # 空前缀
    for num in nums:
        cur_sum += num
        count += prefix_count[cur_sum - k]
        prefix_count[cur_sum] += 1
    return count
```

**ACM模式完整代码**

```python
import sys
from collections import defaultdict
input = sys.stdin.readline

def solve():
    n, k = map(int, input().split())
    nums = list(map(int, input().split()))
    count = 0
    cur_sum = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    for num in nums:
        cur_sum += num
        count += prefix_count[cur_sum - k]
        prefix_count[cur_sum] += 1
    print(count)

solve()
```

---

### 567. 字符串的排列

> 固定长度滑动窗口+字符计数

**M - 暴力解**

```
For each substring of s2 with length len(s1):
  Sort it and compare with sorted s1
Time: O(n * m * log(m)), where n=len(s2), m=len(s1)
Space: O(m)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n * m * log(m)) 排序比较 |
| 下界 | O(n) 至少遍历 s2 一次 |
| 目标 | O(n) time（26为常数）, O(1) space |

**K - 关键词触发**

- "排列/异位词" → 字符频率计数
- "固定长度子串" → 固定长度滑动窗口
- "s1的排列是否在s2中" → 窗口大小 = len(s1)

**E - 优化方案**

固定长度滑动窗口：维护窗口内字符频率数组，与 s1 的频率数组比较。窗口滑动时，加入右端字符、移除左端字符，用一个 diff 计数器追踪不匹配的字符种类数。

- Time: O(n)（26为常数）
- Space: O(1)（固定26个字母）

**核心代码片段**

```python
def checkInclusion(s1, s2):
    if len(s1) > len(s2):
        return False
    count = [0] * 26
    for c in s1:
        count[ord(c) - ord('a')] += 1
    window = [0] * 26
    for i in range(len(s2)):
        window[ord(s2[i]) - ord('a')] += 1
        if i >= len(s1):
            window[ord(s2[i - len(s1)]) - ord('a')] -= 1
        if window == count:
            return True
    return False
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    s1 = input().strip()
    s2 = input().strip()
    m, n = len(s1), len(s2)
    if m > n:
        print("false")
        return
    count = [0] * 26
    window = [0] * 26
    for c in s1:
        count[ord(c) - ord('a')] += 1
    for i in range(n):
        window[ord(s2[i]) - ord('a')] += 1
        if i >= m:
            window[ord(s2[i - m]) - ord('a')] -= 1
        if window == count:
            print("true")
            return
    print("false")

solve()
```

---

### 75. 颜色分类

> 荷兰国旗问题，三指针

**M - 暴力解**

```
Count occurrences of 0, 1, 2
Overwrite array: all 0s, then 1s, then 2s (two-pass)
Time: O(n), Space: O(1)
```

**I - 边界分析**

| 项目 | 值 |
|------|-----|
| 上界 | O(n) 两趟（计数+覆写） |
| 下界 | O(n) 每个元素至少看一次 |
| 目标 | O(n) time, O(1) space, 一趟扫描 |

**K - 关键词触发**

- "三种值/荷兰国旗" → 三指针分区（Dijkstra's three-way partition）
- "原地排序/0-1-2" → lo/mid/hi 三指针

**E - 优化方案**

三指针（荷兰国旗）：
- lo: 下一个0应放的位置
- hi: 下一个2应放的位置
- mid: 当前遍历指针

规则：
- nums[mid]==0: swap(lo, mid), lo++, mid++
- nums[mid]==1: mid++
- nums[mid]==2: swap(mid, hi), hi--（mid不动，因为换来的值未检查）

- Time: O(n)
- Space: O(1)

**核心代码片段**

```python
def sortColors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    nums = list(map(int, input().split()))
    lo, mid, hi = 0, 0, n - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
    print(' '.join(map(str, nums)))

solve()
```
