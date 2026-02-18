---

### Day 18 — 字节高频综合（下）

---

### 10. 正则表达式匹配

**M - 暴力解**
```
Recursive backtracking: try all possible matches
For each char in s, try matching with p considering '.' and '*'
Time: O(2^(m+n)) exponential, Space: O(m+n) recursion stack
```

**I - 边界分析**
- 上界：O(2^(m+n)) 递归回溯所有可能
- 下界：O(m * n) 至少检查每个字符对
- 目标：O(m * n) time, O(m * n) space
- m = len(s), n = len(p)

**K - 关键词触发**
- 字符串匹配 + 复杂规则 → 动态规划
- '.' 匹配任意单个字符 → 简单情况
- '*' 匹配0次或多次前一个字符 → 状态转移分支
- 二维DP：dp[i][j] 表示 s[0:i] 与 p[0:j] 是否匹配

**E - 优化方案**
- dp[i][j] = s前i个字符与p前j个字符是否匹配
- 初始化：dp[0][0] = True，dp[0][j] 处理 a*b*c* 这种可匹配空串的情况
- 转移：
  - 若 p[j-1] 是普通字符或'.'：dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
  - 若 p[j-1] 是'*'：
    - 匹配0次：dp[i][j] = dp[i][j-2]
    - 匹配1+次：dp[i][j] = dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.')
- Time: O(m * n), Space: O(m * n)

**核心代码片段**
```python
def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 初始化：处理 a*b*c* 可匹配空串
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # 匹配0次
                dp[i][j] = dp[i][j - 2]
                # 匹配1+次
                if s[i - 1] == p[j - 2] or p[j - 2] == '.':
                    dp[i][j] |= dp[i - 1][j]
            else:
                # 普通字符或'.'
                if s[i - 1] == p[j - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    s = input().strip()
    p = input().strip()
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]
                if s[i - 1] == p[j - 2] or p[j - 2] == '.':
                    dp[i][j] |= dp[i - 1][j]
            else:
                if s[i - 1] == p[j - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]

    print("true" if dp[m][n] else "false")

solve()
```

---

### 32. 最长有效括号

**M - 暴力解**
```
For each substring, check if it's valid parentheses
Time: O(n^3) enumerate + validate, Space: O(n)
```

**I - 边界分析**
- 上界：O(n^3) 枚举所有子串并验证
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space

**K - 关键词触发**
- 括号匹配 → 栈
- 最长连续有效 → 栈记录下标，计算长度
- 栈底存储"最后一个未匹配的右括号下标"

**E - 优化方案**
- 栈初始化：压入-1作为基准
- 遇'('：下标入栈
- 遇')'：
  - 弹栈
  - 若栈空：说明当前')'无法匹配，将当前下标入栈作为新基准
  - 若栈不空：长度 = 当前下标 - 栈顶下标
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def longestValidParentheses(s: str) -> int:
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    s = input().strip()
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    print(max_len)

solve()
```

---

### 297. 二叉树的序列化与反序列化

**M - 暴力解**
```
Serialize: BFS/DFS traversal, record all nodes including null
Deserialize: Reconstruct tree from traversal
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界/下界：O(n) 必须访问所有节点
- 目标：O(n) time, O(n) space
- 关键：选择合适的遍历方式和null表示

**K - 关键词触发**
- 序列化/反序列化 → 前序/层序遍历 + null标记
- 前序DFS：根-左-右，null用"#"表示
- 层序BFS：逐层，null用"#"表示
- 分隔符：逗号

**E - 优化方案**
- 方案1：前序DFS序列化，递归反序列化
  - serialize: root.val, serialize(left), serialize(right)
  - deserialize: 用迭代器/索引依次读取
- 方案2：层序BFS序列化，队列反序列化
  - serialize: 队列逐层遍历
  - deserialize: 队列重建，每个节点连接两个子节点
- Time: O(n), Space: O(n)

**核心代码片段**
```python
# 前序DFS方案
class Codec:
    def serialize(self, root):
        if not root:
            return "#"
        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"

    def deserialize(self, data):
        def build(vals):
            val = next(vals)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = build(vals)
            node.right = build(vals)
            return node
        return build(iter(data.split(',')))
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize(root):
    if not root:
        return "#"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    def build(vals):
        val = next(vals)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = build(vals)
        node.right = build(vals)
        return node
    return build(iter(data.split(',')))

def solve():
    # 读取层序输入构建树
    line = input().strip()
    if line == "#":
        root = None
    else:
        vals = line.split()
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1
        while queue and i < len(vals):
            node = queue.popleft()
            if vals[i] != "#":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            if i < len(vals) and vals[i] != "#":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1

    # 序列化
    serialized = serialize(root)
    print(serialized)

    # 反序列化
    new_root = deserialize(serialized)
    print(serialize(new_root))

solve()
```

---

### 543. 二叉树的直径

**M - 暴力解**
```
For each node, compute left depth + right depth
Track maximum across all nodes
Time: O(n^2) if recompute depth each time, Space: O(h)
```

**I - 边界分析**
- 上界：O(n^2) 每个节点重复计算深度
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(h) space

**K - 关键词触发**
- 树的直径 = 任意两节点最长路径 → 经过某节点的最长路径 = 左深度 + 右深度
- 后序遍历 → 先计算子树深度，再更新全局最大值
- 全局变量记录最大直径

**E - 优化方案**
- 递归函数返回当前节点的深度，同时更新全局最大直径
- 对每个节点：diameter = left_depth + right_depth
- 返回：max(left_depth, right_depth) + 1
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def diameterOfBinaryTree(root) -> int:
    max_diameter = 0

    def depth(node):
        nonlocal max_diameter
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        max_diameter = max(max_diameter, left + right)
        return max(left, right) + 1

    depth(root)
    return max_diameter
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def solve():
    line = input().strip()
    if not line or line == "#":
        print(0)
        return

    vals = line.split()
    root = TreeNode(int(vals[0]))
    queue = deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        if i < len(vals) and vals[i] != "#":
            node.left = TreeNode(int(vals[i]))
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != "#":
            node.right = TreeNode(int(vals[i]))
            queue.append(node.right)
        i += 1

    max_diameter = 0

    def depth(node):
        nonlocal max_diameter
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        max_diameter = max(max_diameter, left + right)
        return max(left, right) + 1

    depth(root)
    print(max_diameter)

solve()
```

---

### 148. 排序链表

**M - 暴力解**
```
Convert to array, sort, rebuild linked list
Time: O(n log n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n log n) 比较排序下界
- 下界：O(n log n) 必须比较
- 目标：O(n log n) time, O(1) space（原地排序）
- 链表排序 → 归并排序最适合（不需要随机访问）

**K - 关键词触发**
- 链表排序 + O(n log n) + O(1) space → 归并排序（自底向上）
- 快慢指针找中点
- 合并两个有序链表

**E - 优化方案**
- 自顶向下归并（递归）：
  - 快慢指针找中点，断开链表
  - 递归排序左右两半
  - 合并两个有序链表
  - Time: O(n log n), Space: O(log n) 递归栈
- 自底向上归并（迭代）：
  - 从长度1开始，每次合并相邻的两段
  - 长度翻倍，直到覆盖整个链表
  - Time: O(n log n), Space: O(1)

**核心代码片段**
```python
def sortList(head):
    if not head or not head.next:
        return head

    # 快慢指针找中点
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # 断开链表
    mid = slow.next
    slow.next = None

    # 递归排序
    left = sortList(head)
    right = sortList(mid)

    # 合并
    dummy = ListNode()
    cur = dummy
    while left and right:
        if left.val < right.val:
            cur.next = left
            left = left.next
        else:
            cur.next = right
            right = right.next
        cur = cur.next
    cur.next = left if left else right

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

def sortList(head):
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sortList(head)
    right = sortList(mid)

    dummy = ListNode()
    cur = dummy
    while left and right:
        if left.val < right.val:
            cur.next = left
            left = left.next
        else:
            cur.next = right
            right = right.next
        cur = cur.next
    cur.next = left if left else right

    return dummy.next

def solve():
    vals = list(map(int, input().split()))
    if not vals:
        return

    dummy = ListNode()
    cur = dummy
    for val in vals:
        cur.next = ListNode(val)
        cur = cur.next

    sorted_head = sortList(dummy.next)

    result = []
    while sorted_head:
        result.append(str(sorted_head.val))
        sorted_head = sorted_head.next
    print(" ".join(result))

solve()
```

---

### 4. 寻找两个正序数组的中位数

**M - 暴力解**
```
Merge two arrays, find median
Time: O(m + n), Space: O(m + n)
```

**I - 边界分析**
- 上界：O(m + n) 归并
- 下界：O(log(min(m, n))) 二分搜索
- 目标：O(log(min(m, n))) time, O(1) space
- 题目要求 O(log(m+n))，实际可达 O(log(min(m, n)))

**K - 关键词触发**
- 两个有序数组 + O(log) → 二分搜索
- 中位数 → 找第k小元素 → 二分切割位置
- 切割后左半最大 ≤ 右半最小

**E - 优化方案**
- 在较短数组上二分切割位置i（0 到 m）
- 另一数组切割位置 j = (m + n + 1) / 2 - i
- 保证左半元素个数 = 右半元素个数（或多1个）
- 检查条件：
  - nums1[i-1] ≤ nums2[j]
  - nums2[j-1] ≤ nums1[i]
- 若不满足，调整i：
  - nums1[i-1] > nums2[j]：i太大，右移右边界
  - nums2[j-1] > nums1[i]：i太小，左移左边界
- 找到正确切割后：
  - 奇数：max(左半)
  - 偶数：(max(左半) + min(右半)) / 2
- Time: O(log(min(m, n))), Space: O(1)

**核心代码片段**
```python
def findMedianSortedArrays(nums1, nums2) -> float:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i

        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = i - 1
        else:
            left = i + 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums1 = list(map(int, input().split()))
    nums2 = list(map(int, input().split()))

    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i

        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                result = (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                result = max(max_left1, max_left2)
            print(result)
            return
        elif max_left1 > min_right2:
            right = i - 1
        else:
            left = i + 1

solve()
```

---
