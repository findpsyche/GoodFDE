---

### Day 7 — 二叉树（基础）

---

### 144. 二叉树的前序遍历

**M - 暴力解**
```
递归遍历：根→左→右
Time: O(n), Space: O(n) 递归栈
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 已是最优

**K - 关键词触发**
- 前序遍历 → 递归 / 栈迭代

**E - 优化方案**
- 迭代：栈模拟，根入栈→弹出访问→右左子节点依次入栈
- Time: O(n), Space: O(n)

**核心代码片段**
```python
# 递归
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# 迭代
def preorderTraversal(root):
    if not root: return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]
    i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i]))
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i]))
            queue.append(node.right)
        i += 1
    return root

def preorderTraversal(root):
    if not root: return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return res

vals = input().split()
root = build_tree(vals)
print(' '.join(map(str, preorderTraversal(root))))
```

---

### 94. 二叉树的中序遍历

**M - 暴力解**
```
递归遍历：左→根→右
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 中序遍历 → 递归 / 栈迭代（一路向左压栈）

**E - 优化方案**
- 迭代：指针一路向左压栈，弹出访问，转向右子树
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def inorderTraversal(root):
    stack, res = [], []
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def inorderTraversal(root):
    stack, res, cur = [], [], root
    while cur or stack:
        while cur:
            stack.append(cur); cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res

vals = input().split()
root = build_tree(vals)
print(' '.join(map(str, inorderTraversal(root))))
```

---

### 145. 二叉树的后序遍历

**M - 暴力解**
```
递归遍历：左→右→根
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 后序遍历 → 递归 / 前序变体反转（根右左 → 反转得左右根）

**E - 优化方案**
- 迭代：按"根→右→左"遍历，结果反转得到"左→右→根"
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def postorderTraversal(root):
    if not root: return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.left: stack.append(node.left)
        if node.right: stack.append(node.right)
    return res[::-1]
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def postorderTraversal(root):
    if not root: return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.left: stack.append(node.left)
        if node.right: stack.append(node.right)
    return res[::-1]

vals = input().split()
root = build_tree(vals)
print(' '.join(map(str, postorderTraversal(root))))
```

---

### 102. 二叉树的层序遍历

**M - 暴力解**
```
BFS用队列逐层遍历，即为标准解法
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 已是最优

**K - 关键词触发**
- 层序/逐层 → BFS + 队列
- 记录每层 → 队列中记录当前层size

**E - 优化方案**
- BFS：根入队→记录当前层size→弹出size个节点→子节点入队
- Time: O(n), Space: O(n)

**核心代码片段**
```python
from collections import deque

def levelOrder(root):
    if not root: return []
    queue = deque([root])
    res = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def levelOrder(root):
    if not root: return []
    queue = deque([root])
    res = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res

vals = input().split()
root = build_tree(vals)
for level in levelOrder(root):
    print(' '.join(map(str, level)))
```

---

### 226. 翻转二叉树

**M - 暴力解**
```
递归交换每个节点的左右子树
Time: O(n), Space: O(n) 递归栈
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 翻转/镜像 → 递归交换左右子树

**E - 优化方案**
- 递归：先递归翻转左右子树，再交换
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

def print_tree(root):
    if not root: print(""); return
    queue = deque([root]); res = []
    while queue:
        node = queue.popleft()
        if node:
            res.append(str(node.val))
            queue.append(node.left); queue.append(node.right)
        else:
            res.append("null")
    while res and res[-1] == "null": res.pop()
    print(' '.join(res))

vals = input().split()
root = build_tree(vals)
print_tree(invertTree(root))
```

---

### 101. 对称二叉树

**M - 暴力解**
```
翻转右子树，然后比较左子树和翻转后的右子树是否相同
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 对称 → 递归比较：左的左 vs 右的右，左的右 vs 右的左

**E - 优化方案**
- 递归：定义compare(left, right)同时比较两个对称位置
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def isSymmetric(root):
    def compare(left, right):
        if not left and not right: return True
        if not left or not right: return False
        return (left.val == right.val and
                compare(left.left, right.right) and
                compare(left.right, right.left))
    return compare(root.left, root.right) if root else True
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def isSymmetric(root):
    def compare(left, right):
        if not left and not right: return True
        if not left or not right: return False
        return (left.val == right.val and
                compare(left.left, right.right) and
                compare(left.right, right.left))
    return compare(root.left, root.right) if root else True

vals = input().split()
root = build_tree(vals)
print("true" if isSymmetric(root) else "false")
```

---

### 104. 二叉树的最大深度

**M - 暴力解**
```
递归求左右子树深度取较大值+1
Time: O(n), Space: O(h)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 最大深度 → 递归DFS / BFS层数

**E - 优化方案**
- 递归：max(左深度, 右深度) + 1
- BFS：层序遍历计层数
- Time: O(n), Space: O(h) 或 O(n)

**核心代码片段**
```python
def maxDepth(root):
    if not root: return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def maxDepth(root):
    if not root: return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1

vals = input().split()
root = build_tree(vals)
print(maxDepth(root))
```

---

### 199. 二叉树的右视图

**M - 暴力解**
```
BFS层序遍历，取每层最后一个节点
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 右视图/每层最右 → BFS层序遍历取每层末尾

**E - 优化方案**
- BFS：逐层遍历，每层最后一个节点加入结果
- 或DFS：优先访问右子树，每层第一个访问的即为右视图
- Time: O(n), Space: O(n)

**核心代码片段**
```python
from collections import deque

def rightSideView(root):
    if not root: return []
    queue = deque([root])
    res = []
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return res
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def rightSideView(root):
    if not root: return []
    queue = deque([root])
    res = []
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return res

vals = input().split()
root = build_tree(vals)
print(' '.join(map(str, rightSideView(root))))
```

---

### 110. 平衡二叉树

**M - 暴力解**
```
对每个节点分别求左右子树高度，检查差值
Time: O(n^2)（每个节点都重新算高度）, Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 平衡判断 → 自底向上递归，计算高度同时判断

**E - 优化方案**
- 自底向上：递归返回高度，不平衡时返回-1提前终止
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def isBalanced(root):
    def height(node):
        if not node: return 0
        left = height(node.left)
        if left == -1: return -1
        right = height(node.right)
        if right == -1: return -1
        if abs(left - right) > 1: return -1
        return max(left, right) + 1
    return height(root) != -1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def build_tree(vals):
    if not vals or vals[0] == 'null': return None
    root = TreeNode(int(vals[0]))
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i])); queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i])); queue.append(node.right)
        i += 1
    return root

def isBalanced(root):
    def height(node):
        if not node: return 0
        left = height(node.left)
        if left == -1: return -1
        right = height(node.right)
        if right == -1: return -1
        if abs(left - right) > 1: return -1
        return max(left, right) + 1
    return height(root) != -1

vals = input().split()
root = build_tree(vals)
print("true" if isBalanced(root) else "false")
```
