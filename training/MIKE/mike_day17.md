---

### Day 17 — 字节高频综合（上）

---

### 215. 数组中的第K个最大元素

**M - 暴力解**
```
Sort array in descending order, return nums[k-1]
Time: O(n log n), Space: O(1)
```

**I - 边界分析**
- 上界：O(n log n) 排序
- 下界：O(n) 至少遍历一次
- 目标：O(n) 平均时间（快速选择）或 O(n log k)（堆）
- k保证有效，1 <= k <= n

**K - 关键词触发**
- 第K个最大/最小 → 小顶堆（维护k个最大）/ 快速选择算法
- 不需要完全排序 → 快速选择partition，只递归一侧
- 堆优化 → 维护大小为k的小顶堆，堆顶即第k大

**E - 优化方案**
- 方案1：快速选择（QuickSelect），partition分区后比较pivot位置与目标位置，只递归一侧。平均O(n)，最坏O(n^2)
- 方案2：小顶堆，维护k个最大元素，遍历数组时若元素>堆顶则替换。Time: O(n log k), Space: O(k)
- 字节面试推荐快速选择（展示算法功底）

**核心代码片段**
```python
import random

def findKthLargest(nums: list, k: int) -> int:
    def partition(l, r):
        pivot_idx = random.randint(l, r)
        nums[pivot_idx], nums[r] = nums[r], nums[pivot_idx]
        pivot = nums[r]
        i = l
        for j in range(l, r):
            if nums[j] >= pivot:  # 降序partition
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    l, r = 0, len(nums) - 1
    target = k - 1
    while True:
        pos = partition(l, r)
        if pos == target:
            return nums[pos]
        elif pos < target:
            l = pos + 1
        else:
            r = pos - 1
```

**ACM模式完整代码**
```python
import sys
import random
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    k = int(input())

    def partition(l, r):
        pivot_idx = random.randint(l, r)
        nums[pivot_idx], nums[r] = nums[r], nums[pivot_idx]
        pivot = nums[r]
        i = l
        for j in range(l, r):
            if nums[j] >= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    l, r = 0, len(nums) - 1
    target = k - 1
    while True:
        pos = partition(l, r)
        if pos == target:
            print(nums[pos])
            return
        elif pos < target:
            l = pos + 1
        else:
            r = pos - 1

solve()
```

---

### 146. LRU缓存

**M - 暴力解**
```
Use list to track access order, dict for key-value storage
On get: move to end, on put: append and remove oldest if over capacity
Time: O(n) for list operations, Space: O(capacity)
```

**I - 边界分析**
- 上界：O(n) 使用列表维护顺序
- 下界：O(1) 理想情况
- 目标：get和put都是O(1)
- 需要快速查找(哈希表) + 快速删除/插入(双向链表)

**K - 关键词触发**
- LRU缓存 → 哈希表 + 双向链表
- O(1) get/put → 哈希表定位节点，双向链表调整顺序
- 最近使用 → 移到链表头部，最久未使用 → 删除链表尾部

**E - 优化方案**
- 哈希表{key: Node} + 双向链表（带dummy head/tail）
- get(key): 若存在，移到头部，返回value；否则返回-1
- put(key, value): 若存在，更新value并移到头部；否则新建节点插头部，超容量则删尾部节点
- Time: O(1), Space: O(capacity)

**核心代码片段**
```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            if len(self.cache) > self.cap:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            if len(self.cache) > self.cap:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]

def solve():
    capacity = int(input())
    lru = LRUCache(capacity)
    n = int(input())
    for _ in range(n):
        op = input().split()
        if op[0] == "get":
            print(lru.get(int(op[1])))
        else:
            lru.put(int(op[1]), int(op[2]))

solve()
```

---

### 460. LFU缓存

**M - 暴力解**
```
Use list to track frequency, dict for key-value storage
On get/put: update frequency, linear search for LFU
Time: O(n) for list operations, Space: O(capacity)
```

**I - 边界分析**
- 上界：O(n) 使用列表维护频次桶
- 下界：O(1) 理想情况
- 目标：get和put都是O(1)
- 需要维护频次 + 同频次内的LRU顺序

**K - 关键词触发**
- LFU缓存 → 双哈希表 + 频次链表
- 最少使用频次 → 维护minFreq变量
- 同频次LRU → 每个频次对应一个双向链表
- O(1)操作 → key→node映射 + freq→链表映射

**E - 优化方案**
- cache: {key: Node(key, val, freq)}
- freq_map: {freq: 双向链表}
- minFreq: 当前最小频次
- get(key): 若存在，freq++，从旧频次链表移到新频次链表，更新minFreq
- put(key, value): 若存在，更新value并增加频次；否则新建freq=1节点，超容量则删除minFreq链表尾部
- Time: O(1), Space: O(capacity)

**核心代码片段**
```python
class Node:
    def __init__(self, key=0, val=0, freq=0):
        self.key = key
        self.val = val
        self.freq = freq
        self.prev = None
        self.next = None

class DLinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_first(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def remove_last(self):
        if self.is_empty():
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def is_empty(self):
        return self.head.next == self.tail

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq_map = {}
        self.min_freq = 0

    def _update_freq(self, node):
        freq = node.freq
        self.freq_map[freq].remove(node)
        if self.freq_map[freq].is_empty():
            del self.freq_map[freq]
            if freq == self.min_freq:
                self.min_freq += 1

        node.freq += 1
        if node.freq not in self.freq_map:
            self.freq_map[node.freq] = DLinkedList()
        self.freq_map[node.freq].add_first(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._update_freq(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._update_freq(node)
        else:
            if len(self.cache) >= self.capacity:
                lfu_list = self.freq_map[self.min_freq]
                lfu_node = lfu_list.remove_last()
                del self.cache[lfu_node.key]

            node = Node(key, value, 1)
            self.cache[key] = node
            if 1 not in self.freq_map:
                self.freq_map[1] = DLinkedList()
            self.freq_map[1].add_first(node)
            self.min_freq = 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class Node:
    def __init__(self, key=0, val=0, freq=0):
        self.key = key
        self.val = val
        self.freq = freq
        self.prev = None
        self.next = None

class DLinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_first(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def remove_last(self):
        if self.is_empty():
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def is_empty(self):
        return self.head.next == self.tail

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq_map = {}
        self.min_freq = 0

    def _update_freq(self, node):
        freq = node.freq
        self.freq_map[freq].remove(node)
        if self.freq_map[freq].is_empty():
            del self.freq_map[freq]
            if freq == self.min_freq:
                self.min_freq += 1

        node.freq += 1
        if node.freq not in self.freq_map:
            self.freq_map[node.freq] = DLinkedList()
        self.freq_map[node.freq].add_first(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._update_freq(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._update_freq(node)
        else:
            if len(self.cache) >= self.capacity:
                lfu_list = self.freq_map[self.min_freq]
                lfu_node = lfu_list.remove_last()
                del self.cache[lfu_node.key]

            node = Node(key, value, 1)
            self.cache[key] = node
            if 1 not in self.freq_map:
                self.freq_map[1] = DLinkedList()
            self.freq_map[1].add_first(node)
            self.min_freq = 1

def solve():
    capacity = int(input())
    lfu = LFUCache(capacity)
    n = int(input())
    for _ in range(n):
        op = input().split()
        if op[0] == "get":
            print(lfu.get(int(op[1])))
        else:
            lfu.put(int(op[1]), int(op[2]))

solve()
```

---

### 23. 合并K个升序链表

**M - 暴力解**
```
Merge lists one by one: merge(list1, list2), then merge result with list3, etc.
Time: O(k * n * k) where n = avg list length, Space: O(1)
```

**I - 边界分析**
- 上界：O(k^2 * n) 逐个合并
- 下界：O(n * k) 至少遍历所有节点
- 目标：O(n * k * log k) 使用堆
- k个链表，总共n*k个节点

**K - 关键词触发**
- 合并K个有序序列 → 最小堆（优先队列）
- K路归并 → 堆维护k个候选，每次取最小
- 分治思想 → 两两合并，类似归并排序

**E - 优化方案**
- 方案1：最小堆，将k个链表头节点入堆，每次弹出最小节点接入结果，将其next入堆。Time: O(n*k*log k), Space: O(k)
- 方案2：分治合并，两两配对递归合并。Time: O(n*k*log k), Space: O(log k)递归栈
- 字节面试推荐堆方案（更直观）

**核心代码片段**
```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: list) -> ListNode:
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode()
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**ACM模式完整代码**
```python
import sys
import heapq
input = sys.stdin.readline

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def solve():
    k = int(input())
    lists = []
    for _ in range(k):
        arr = list(map(int, input().split()))
        if not arr:
            lists.append(None)
            continue
        head = ListNode(arr[0])
        cur = head
        for val in arr[1:]:
            cur.next = ListNode(val)
            cur = cur.next
        lists.append(head)

    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode()
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    res = []
    cur = dummy.next
    while cur:
        res.append(str(cur.val))
        cur = cur.next
    print(" ".join(res))

solve()
```

---

### 33. 搜索旋转排序数组

**M - 暴力解**
```
Linear scan to find target
Time: O(n), Space: O(1)
```

**I - 边界分析**
- 上界：O(n) 线性扫描
- 下界：O(log n) 二分查找
- 目标：O(log n) time, O(1) space
- 数组无重复，旋转后分为两段有序

**K - 关键词触发**
- 旋转排序数组 → 二分查找变种
- 判断有序半边 → 比较nums[mid]与nums[left]
- target在有序半边 → 正常二分，否则搜索另一半

**E - 优化方案**
- 二分查找，每次判断左半[left, mid]或右半[mid, right]哪边有序
- 若左半有序且target在[nums[left], nums[mid]]，搜索左半；否则搜索右半
- 若右半有序且target在[nums[mid], nums[right]]，搜索右半；否则搜索左半
- Time: O(log n), Space: O(1)

**核心代码片段**
```python
def search(nums: list, target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:  # 左半有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    target = int(input())

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            print(mid)
            return

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    print(-1)

solve()
```

---

### 31. 下一个排列

**M - 暴力解**
```
Generate all permutations, sort, find current and return next
Time: O(n! * n log n), Space: O(n!)
```

**I - 边界分析**
- 上界：O(n!) 生成所有排列
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space 原地修改
- 字典序下一个排列

**K - 关键词触发**
- 下一个排列 → 从后往前找第一个升序对
- 字典序最小增长 → 找到交换位置后反转后缀
- 原地修改 → 不使用额外空间

**E - 优化方案**
- 从后往前找第一个nums[i] < nums[i+1]的位置i（第一个"较小数"）
- 从后往前找第一个nums[j] > nums[i]的位置j（第一个"较大数"）
- 交换nums[i]和nums[j]
- 反转i+1到末尾的子数组（使其升序，保证字典序最小）
- 若找不到i（整个数组降序），反转整个数组
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def nextPermutation(nums: list) -> None:
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        j = n - 1
        while j > i and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    # 反转i+1到末尾
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        j = n - 1
        while j > i and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

    print(" ".join(map(str, nums)))

solve()
```

---

### 128. 最长连续序列

**M - 暴力解**
```
For each number, check consecutive numbers exist by linear scan
Time: O(n^3), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^3) 暴力枚举起点和长度
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space
- 不能排序（要求O(n)）

**K - 关键词触发**
- 连续序列 → 哈希集合快速查找
- O(n)时间 → 只从序列起点开始计数
- 序列起点 → num-1不在集合中

**E - 优化方案**
- 将所有数放入哈希集合
- 遍历集合，对每个num，只有当num-1不在集合时才作为起点
- 从起点开始向后查找num+1, num+2...，统计长度
- 每个数最多被访问2次（一次遍历，一次作为序列成员），Time: O(n), Space: O(n)

**核心代码片段**
```python
def longestConsecutive(nums: list) -> int:
    num_set = set(nums)
    max_len = 0

    for num in num_set:
        if num - 1 not in num_set:  # 序列起点
            cur = num
            cur_len = 1
            while cur + 1 in num_set:
                cur += 1
                cur_len += 1
            max_len = max(max_len, cur_len)

    return max_len
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    if not nums:
        print(0)
        return

    num_set = set(nums)
    max_len = 0

    for num in num_set:
        if num - 1 not in num_set:
            cur = num
            cur_len = 1
            while cur + 1 in num_set:
                cur += 1
                cur_len += 1
            max_len = max(max_len, cur_len)

    print(max_len)

solve()
```

---

### 41. 缺失的第一个正数

**M - 暴力解**
```
Use set to store all numbers, iterate from 1 until find missing
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n) + O(n) space 使用哈希集合
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space
- 答案范围[1, n+1]

**K - 关键词触发**
- O(1)空间 → 原地哈希，利用数组本身作为哈希表
- 缺失的正数 → 将值i放到下标i-1位置
- 范围[1, n] → 负数、0、>n的数都忽略

**E - 优化方案**
- 原地哈希：将每个正数nums[i]放到下标nums[i]-1的位置
- 遍历数组，对于1 <= nums[i] <= n且nums[i] != nums[nums[i]-1]，交换
- 再次遍历，找第一个nums[i] != i+1的位置，返回i+1
- 若都匹配，返回n+1
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def firstMissingPositive(nums: list) -> int:
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            idx = nums[i] - 1
            nums[i], nums[idx] = nums[idx], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    n = len(nums)

    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            idx = nums[i] - 1
            nums[i], nums[idx] = nums[idx], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            print(i + 1)
            return
    print(n + 1)

solve()
```

---
