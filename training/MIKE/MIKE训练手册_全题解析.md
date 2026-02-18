# MIKE训练手册 — 20天算法冲刺全题解析

> 配套《20天算法冲刺计划》使用
> 每道题包含：MIKE四步分析 + 核心代码片段 + ACM模式Python完整代码

---

## 目录

- [Phase 1：基础数据结构（Day 1-5）](#phase-1基础数据结构day-1-5)
- [Phase 2：核心算法（Day 6-10）](#phase-2核心算法day-6-10)
- [Phase 3：高级算法（Day 11-   16）](#phase-3高级算法day-11-16)
- [Phase 4：模拟冲刺（Day 17-20）](#phase-4模拟冲刺day-17-20)

---

## Phase 1：基础数据结构（Day 1-5）

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

**I - 边界分析**
- 上界：O(n)（线性扫描）
- 下界：O(log n)（有序数组查找的理论下界）
- 目标：O(log n)

**K - 关键词触发**
- "有序数组" → 二分查找
- "查找目标值" → 二分查找

**E - 优化方案**
- 二分查找：每次比较中间值，缩小一半搜索范围
- 时间：O(log n)  空间：O(1)

**核心代码片段**
```python
left, right = 0, len(nums) - 1
while left <= right:
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

---

### Day 2 — 链表

---

### 203. 移除链表元素

**M - 暴力解**
```
遍历链表，遇到值等于val的节点就删除
需要特殊处理头节点

时间：O(n)  空间：O(1)
```

**I - 边界分析**
- 上界 = 下界 = O(n)（必须遍历每个节点）
- 已是最优

**K - 关键词触发**
- "删除链表节点" → 虚拟头节点统一处理

**E - 优化方案**
- 虚拟头节点：避免头节点特殊处理
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
dummy = ListNode(0, head)
cur = dummy
while cur.next:
    if cur.next.val == val:
        cur.next = cur.next.next
    else:
        cur = cur.next
return dummy.next
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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
    print(' '.join(res) if res else "")

def remove_elements(head, val):
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next

nums = list(map(int, input().split()))
val = int(input())
head = build_list(nums)
print_list(remove_elements(head, val))
```

---

### 707. 设计链表

**M - 暴力解**
```
用数组模拟链表操作
get: 按下标访问 O(1)
add/delete: 需要移动元素 O(n)

时间：O(n)  空间：O(n)
```

**I - 边界分析**
- get: 下界O(n)（单链表必须遍历）
- add/delete: 下界O(n)（需要定位前驱）

**K - 关键词触发**
- "设计链表" → 虚拟头节点 + size计数

**E - 优化方案**
- 虚拟头节点简化边界处理，size变量O(1)获取长度
- 时间：get/add/delete均O(n)  空间：O(n)

**核心代码片段**
```python
class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode(0)
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size: return -1
        cur = self.dummy.next
        for _ in range(index): cur = cur.next
        return cur.val

    def addAtIndex(self, index, val):
        if index > self.size: return
        index = max(0, index)
        cur = self.dummy
        for _ in range(index): cur = cur.next
        cur.next = ListNode(val, cur.next)
        self.size += 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode(0)
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        cur = self.dummy.next
        for _ in range(index):
            cur = cur.next
        return cur.val

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index > self.size:
            return
        index = max(0, index)
        cur = self.dummy
        for _ in range(index):
            cur = cur.next
        cur.next = ListNode(val, cur.next)
        self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        cur = self.dummy
        for _ in range(index):
            cur = cur.next
        cur.next = cur.next.next
        self.size -= 1

# 操作示例
obj = MyLinkedList()
n = int(input())
for _ in range(n):
    line = input().split()
    op = line[0]
    if op == "get":
        print(obj.get(int(line[1])))
    elif op == "addAtHead":
        obj.addAtHead(int(line[1]))
    elif op == "addAtTail":
        obj.addAtTail(int(line[1]))
    elif op == "addAtIndex":
        obj.addAtIndex(int(line[1]), int(line[2]))
    elif op == "deleteAtIndex":
        obj.deleteAtIndex(int(line[1]))
```

---

### 206. 反转链表

**M - 暴力解**
```
将所有节点值存入数组，反转数组，重建链表

时间：O(n)  空间：O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 优化空间到O(1)

**K - 关键词触发**
- "反转链表" → 双指针迭代 / 递归

**E - 优化方案**
- 双指针：prev和cur逐步翻转next指针
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
prev, cur = None, head
while cur:
    nxt = cur.next
    cur.next = prev
    prev = cur
    cur = nxt
return prev
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

nums = list(map(int, input().split()))
head = build_list(nums)
print_list(reverse_list(head))
```

---

### 24. 两两交换链表中的节点

**M - 暴力解**
```
遍历链表，每次取两个节点交换值

时间：O(n)  空间：O(1)
（但交换值不是真正的节点交换）
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- "两两交换" → 虚拟头节点 + 模拟指针操作

**E - 优化方案**
- 虚拟头节点，每次操作三个指针完成一对交换
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
dummy = ListNode(0, head)
cur = dummy
while cur.next and cur.next.next:
    a, b = cur.next, cur.next.next
    cur.next = b
    a.next = b.next
    b.next = a
    cur = a
return dummy.next
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def swap_pairs(head):
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next and cur.next.next:
        a, b = cur.next, cur.next.next
        cur.next = b
        a.next = b.next
        b.next = a
        cur = a
    return dummy.next

nums = list(map(int, input().split()))
head = build_list(nums)
print_list(swap_pairs(head))
```

---

### 19. 删除链表的倒数第N个节点

**M - 暴力解**
```
先遍历一遍算长度L
再遍历到第L-n个节点删除

时间：O(2n)  空间：O(1)
```

**I - 边界分析**
- 上界：O(2n)
- 下界：O(n)
- 目标：一次遍历O(n)

**K - 关键词触发**
- "倒数第N个" → 快慢指针（快指针先走N步）

**E - 优化方案**
- 快慢指针：快指针先走n+1步，然后同步移动
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
dummy = ListNode(0, head)
fast = slow = dummy
for _ in range(n + 1):
    fast = fast.next
while fast:
    fast = fast.next
    slow = slow.next
slow.next = slow.next.next
return dummy.next
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next

nums = list(map(int, input().split()))
n = int(input())
head = build_list(nums)
print_list(remove_nth_from_end(head, n))
```

---

### 160. 链表相交

**M - 暴力解**
```
对A链表每个节点，遍历B链表查找是否相同

时间：O(m·n)  空间：O(1)
```

**I - 边界分析**
- 上界：O(m·n)
- 下界：O(m+n)
- 目标：O(m+n)

**K - 关键词触发**
- "链表相交" → 双指针等距法（走完自己走对方）

**E - 优化方案**
- 双指针：pA走完A走B，pB走完B走A，消除长度差
- 时间：O(m+n)  空间：O(1)

**核心代码片段**
```python
pA, pB = headA, headB
while pA != pB:
    pA = pA.next if pA else headB
    pB = pB.next if pB else headA
return pA
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def get_intersection_node(headA, headB):
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    return pA

# ACM输入：两个链表长度和公共部分
# 格式：lenA lenB skipA skipB intersectVal
# 简化版：两个数组 + 交点位置
la, lb, skip = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
# 构建：a的前skip个独立，后面共享
nodes_a = [ListNode(v) for v in a]
nodes_b = [ListNode(v) for v in b[:skip]]
for i in range(len(nodes_a)-1): nodes_a[i].next = nodes_a[i+1]
for i in range(len(nodes_b)-1): nodes_b[i].next = nodes_b[i+1]
# 简化输出
node = get_intersection_node(nodes_a[0] if nodes_a else None, nodes_b[0] if nodes_b else None)
print(node.val if node else "null")
```

---

### 142. 环形链表 II

**M - 暴力解**
```
用哈希set记录访问过的节点
第一个重复访问的就是环入口

时间：O(n)  空间：O(n)
```

**I - 边界分析**
- 上界：O(n) 时间 O(n) 空间
- 下界：O(n) 时间
- 目标：O(n) 时间 O(1) 空间

**K - 关键词触发**
- "环" + "入口" → 快慢指针 + 数学推导

**E - 优化方案**
- 快慢指针相遇后，一个回到头部，两个同速前进再次相遇即入口
- 数学推导：a = c（头到入口距离 = 相遇点到入口距离）
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
return None
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None

# 输入：数组 + 环入口位置pos（-1表示无环）
nums = list(map(int, input().split()))
pos = int(input())
nodes = [ListNode(v) for v in nums]
for i in range(len(nodes)-1):
    nodes[i].next = nodes[i+1]
if pos >= 0:
    nodes[-1].next = nodes[pos]
result = detect_cycle(nodes[0] if nodes else None)
print(result.val if result else "null")
```

---

### 25. K个一组翻转链表

**M - 暴力解**
```
将链表值存入数组
每K个一组反转数组段
重建链表

时间：O(n)  空间：O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 优化空间到O(1)

**K - 关键词触发**
- "K个一组翻转" → 分组 + 反转 + 重新连接

**E - 优化方案**
- 先计数够K个则反转该段，不够保持原序
- 时间：O(n)  空间：O(1)

**核心代码片段**
```python
dummy = ListNode(0, head)
prev_group = dummy
while True:
    # 检查是否有K个节点
    kth = prev_group
    for _ in range(k):
        kth = kth.next
        if not kth: return dummy.next
    next_group = kth.next
    # 反转K个节点
    prev, cur = next_group, prev_group.next
    for _ in range(k):
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    # 连接
    tmp = prev_group.next
    prev_group.next = prev
    prev_group = tmp
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    prev_group = dummy
    while True:
        kth = prev_group
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        next_group = kth.next
        prev, cur = next_group, prev_group.next
        for _ in range(k):
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        tmp = prev_group.next
        prev_group.next = prev
        prev_group = tmp
    return dummy.next

nums = list(map(int, input().split()))
k = int(input())
head = build_list(nums)
print_list(reverse_k_group(head, k))
```
