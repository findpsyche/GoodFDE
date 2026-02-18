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
la, lb, skip = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
nodes_a = [ListNode(v) for v in a]
nodes_b = [ListNode(v) for v in b[:skip]]
for i in range(len(nodes_a)-1): nodes_a[i].next = nodes_a[i+1]
for i in range(len(nodes_b)-1): nodes_b[i].next = nodes_b[i+1]
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
