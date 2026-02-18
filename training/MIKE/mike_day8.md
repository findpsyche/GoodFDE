---

### Day 8 — 二叉树（进阶）

---

### 700. 二叉搜索树中的搜索

**M - 暴力解**
```
遍历整棵树查找目标值
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n)（退化为链表）
- 下界：O(log n)（平衡BST）
- 目标：O(h)

**K - 关键词触发**
- BST + 查找 → 利用BST性质，小往左大往右

**E - 优化方案**
- BST性质搜索：比当前小往左，比当前大往右
- Time: O(h), Space: O(h) 递归 / O(1) 迭代

**核心代码片段**
```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)
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

def searchBST(root, val):
    while root:
        if root.val == val: return root
        elif val < root.val: root = root.left
        else: root = root.right
    return None

vals = input().split()
root = build_tree(vals)
val = int(input())
node = searchBST(root, val)
print(node.val if node else "null")
```

---

### 98. 验证二叉搜索树

**M - 暴力解**
```
中序遍历存入数组，检查数组是否严格递增
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 优化空间到O(h)

**K - 关键词触发**
- 验证BST → 中序遍历严格递增
- BST性质 → 每个节点有上下界约束

**E - 优化方案**
- 中序遍历时维护prev，比较当前值>prev即可
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def isValidBST(root):
    prev = [float('-inf')]
    def inorder(node):
        if not node: return True
        if not inorder(node.left): return False
        if node.val <= prev[0]: return False
        prev[0] = node.val
        return inorder(node.right)
    return inorder(root)
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

def isValidBST(root):
    prev = [float('-inf')]
    def inorder(node):
        if not node: return True
        if not inorder(node.left): return False
        if node.val <= prev[0]: return False
        prev[0] = node.val
        return inorder(node.right)
    return inorder(root)

vals = input().split()
root = build_tree(vals)
print("true" if isValidBST(root) else "false")
```

---

### 530. 二叉搜索树的最小绝对差

**M - 暴力解**
```
中序遍历存入数组，遍历相邻元素求最小差
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- BST + 最小差 → 中序遍历有序，最小差在相邻节点间

**E - 优化方案**
- 中序遍历时维护prev，实时计算差值
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def getMinimumDifference(root):
    prev, min_diff = None, float('inf')
    def inorder(node):
        nonlocal prev, min_diff
        if not node: return
        inorder(node.left)
        if prev is not None:
            min_diff = min(min_diff, node.val - prev)
        prev = node.val
        inorder(node.right)
    inorder(root)
    return min_diff
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

def getMinimumDifference(root):
    prev, min_diff = None, float('inf')
    def inorder(node):
        nonlocal prev, min_diff
        if not node: return
        inorder(node.left)
        if prev is not None:
            min_diff = min(min_diff, node.val - prev)
        prev = node.val
        inorder(node.right)
    inorder(root)
    return min_diff

vals = input().split()
root = build_tree(vals)
print(getMinimumDifference(root))
```

---

### 236. 二叉树的最近公共祖先

**M - 暴力解**
```
分别记录从根到p和q的路径，找最后一个公共节点
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)

**K - 关键词触发**
- 公共祖先 → 后序遍历（自底向上）
- 两个节点分布 → 左右子树分别查找

**E - 优化方案**
- 后序递归：左右子树分别找p和q，都找到则当前为LCA
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right: return root
    return left if left else right
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
    nodes = {}
    root = TreeNode(int(vals[0]))
    nodes[int(vals[0])] = root
    queue = [root]; i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] != 'null':
            node.left = TreeNode(int(vals[i]))
            nodes[int(vals[i])] = node.left
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != 'null':
            node.right = TreeNode(int(vals[i]))
            nodes[int(vals[i])] = node.right
            queue.append(node.right)
        i += 1
    return root, nodes

def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right: return root
    return left if left else right

vals = input().split()
pv, qv = map(int, input().split())
root, nodes = build_tree(vals)
result = lowestCommonAncestor(root, nodes[pv], nodes[qv])
print(result.val if result else "null")
```

---

### 105. 从前序与中序遍历序列构造二叉树

**M - 暴力解**
```
前序第一个为根，在中序中线性查找根的位置来划分左右子树
Time: O(n^2)（每次线性查找）, Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 前序+中序构造 → 递归分治
- 加速查找 → 哈希表存中序下标

**E - 优化方案**
- 哈希表存中序值→下标映射，O(1)定位根在中序的位置
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def buildTree(preorder, inorder):
    idx_map = {v: i for i, v in enumerate(inorder)}
    pre_idx = [0]

    def build(in_left, in_right):
        if in_left > in_right: return None
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        idx = idx_map[root_val]
        root.left = build(in_left, idx - 1)
        root.right = build(idx + 1, in_right)
        return root

    return build(0, len(inorder) - 1)
```

**ACM模式完整代码**
```python
import sys
from collections import deque
input = sys.stdin.readline

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def buildTree(preorder, inorder):
    idx_map = {v: i for i, v in enumerate(inorder)}
    pre_idx = [0]
    def build(in_left, in_right):
        if in_left > in_right: return None
        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)
        idx = idx_map[root_val]
        root.left = build(in_left, idx - 1)
        root.right = build(idx + 1, in_right)
        return root
    return build(0, len(inorder) - 1)

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

preorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))
print_tree(buildTree(preorder, inorder))
```

---

### 654. 最大二叉树

**M - 暴力解**
```
找区间最大值作为根，左右区间递归构造子树
Time: O(n^2)（每次线性找最大值）, Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n log n)（平衡情况）
- 平均：O(n log n)

**K - 关键词触发**
- 最大值为根+递归构造 → 分治

**E - 优化方案**
- 分治递归，每层找最大值O(n)，共O(log n)层（平均）
- 进阶：单调栈O(n)构造
- Time: O(n log n) 平均, Space: O(n)

**核心代码片段**
```python
def constructMaximumBinaryTree(nums):
    if not nums: return None
    max_idx = nums.index(max(nums))
    root = TreeNode(nums[max_idx])
    root.left = constructMaximumBinaryTree(nums[:max_idx])
    root.right = constructMaximumBinaryTree(nums[max_idx + 1:])
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

def constructMaximumBinaryTree(nums):
    if not nums: return None
    max_idx = nums.index(max(nums))
    root = TreeNode(nums[max_idx])
    root.left = constructMaximumBinaryTree(nums[:max_idx])
    root.right = constructMaximumBinaryTree(nums[max_idx + 1:])
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

nums = list(map(int, input().split()))
print_tree(constructMaximumBinaryTree(nums))
```

---

### 114. 二叉树展开为链表

**M - 暴力解**
```
前序遍历存入数组，重建为只有右子节点的链表
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 优化空间到O(1)

**K - 关键词触发**
- 展开为链表+前序顺序 → 前序遍历 / 寻找前驱节点

**E - 优化方案**
- O(1)空间：对每个节点，将左子树的最右节点连接到右子树，然后左子树移到右边
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def flatten(root):
    cur = root
    while cur:
        if cur.left:
            # 找左子树最右节点
            pre = cur.left
            while pre.right:
                pre = pre.right
            pre.right = cur.right
            cur.right = cur.left
            cur.left = None
        cur = cur.right
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

def flatten(root):
    cur = root
    while cur:
        if cur.left:
            pre = cur.left
            while pre.right:
                pre = pre.right
            pre.right = cur.right
            cur.right = cur.left
            cur.left = None
        cur = cur.right

vals = input().split()
root = build_tree(vals)
flatten(root)
res = []
while root:
    res.append(str(root.val))
    root = root.right
print(' '.join(res))
```

---

### 124. 二叉树中的最大路径和

**M - 暴力解**
```
枚举所有路径计算和，取最大值
Time: O(n^2), Space: O(n)
```

**I - 边界分析**
- 上界：O(n^2)
- 下界：O(n)
- 目标：O(n)

**K - 关键词触发**
- 树中任意路径最大和 → 后序递归，每个节点计算经过自己的最大路径

**E - 优化方案**
- 后序递归：每个节点返回单侧最大贡献（负数取0），同时更新全局最大值
- Time: O(n), Space: O(h)

**核心代码片段**
```python
def maxPathSum(root):
    max_sum = [float('-inf')]
    def dfs(node):
        if not node: return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        max_sum[0] = max(max_sum[0], left + right + node.val)
        return max(left, right) + node.val
    dfs(root)
    return max_sum[0]
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

def maxPathSum(root):
    max_sum = [float('-inf')]
    def dfs(node):
        if not node: return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        max_sum[0] = max(max_sum[0], left + right + node.val)
        return max(left, right) + node.val
    dfs(root)
    return max_sum[0]

vals = input().split()
root = build_tree(vals)
print(maxPathSum(root))
```
