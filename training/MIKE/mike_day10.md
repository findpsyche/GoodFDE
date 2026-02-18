---

### Day 10 — 回溯进阶 + Phase 2 测试

---

### 131. 分割回文串

**M - 暴力解**
```
枚举所有切割方案，检查每段是否回文
Time: O(2^n * n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n * n)（n-1个位置选择切/不切）
- 目标：回溯+回文判断剪枝

**K - 关键词触发**
- 切割+回文 → 回溯，从startIndex尝试切割[startIndex,i]，判断是否回文
- 剪枝 → 非回文直接跳过

**E - 优化方案**
- 回溯+回文判断：从startIndex开始，尝试每个结束位置，是回文则递归
- 优化：预处理DP表记录所有子串是否回文
- Time: O(2^n * n), Space: O(n^2)

**核心代码片段**
```python
def partition(s):
    res = []
    def isPalindrome(sub):
        return sub == sub[::-1]

    def backtrack(start, path):
        if start == len(s):
            res.append(path[:])
            return
        for i in range(start, len(s)):
            if isPalindrome(s[start:i+1]):
                path.append(s[start:i+1])
                backtrack(i + 1, path)
                path.pop()
    backtrack(0, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def partition(s):
    res = []
    def isPalindrome(sub):
        return sub == sub[::-1]

    def backtrack(start, path):
        if start == len(s):
            res.append(path[:])
            return
        for i in range(start, len(s)):
            if isPalindrome(s[start:i+1]):
                path.append(s[start:i+1])
                backtrack(i + 1, path)
                path.pop()
    backtrack(0, [])
    return res

s = input().strip()
for part in partition(s):
    print(' '.join(part))
```

---

### 93. 复原IP地址

**M - 暴力解**
```
枚举所有3个分割点位置，检查4段是否合法
Time: O(n^3), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^3)（3个分割点）
- 目标：回溯+剪枝（段数、长度、值范围）

**K - 关键词触发**
- 切割+约束条件 → 回溯，每段1-3位，值0-255，无前导零
- 4段用完字符串 → 收集结果

**E - 优化方案**
- 回溯+多重剪枝：段数=4且用完字符串才收集，每段检查合法性
- Time: O(3^4) = O(81), Space: O(1)

**核心代码片段**
```python
def restoreIpAddresses(s):
    res = []
    def isValid(seg):
        if not seg or len(seg) > 3: return False
        if seg[0] == '0' and len(seg) > 1: return False
        return 0 <= int(seg) <= 255

    def backtrack(start, path):
        if len(path) == 4:
            if start == len(s):
                res.append('.'.join(path))
            return
        for i in range(start, min(start + 3, len(s))):
            seg = s[start:i+1]
            if isValid(seg):
                path.append(seg)
                backtrack(i + 1, path)
                path.pop()
    backtrack(0, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def restoreIpAddresses(s):
    res = []
    def isValid(seg):
        if not seg or len(seg) > 3: return False
        if seg[0] == '0' and len(seg) > 1: return False
        return 0 <= int(seg) <= 255

    def backtrack(start, path):
        if len(path) == 4:
            if start == len(s):
                res.append('.'.join(path))
            return
        for i in range(start, min(start + 3, len(s))):
            seg = s[start:i+1]
            if isValid(seg):
                path.append(seg)
                backtrack(i + 1, path)
                path.pop()
    backtrack(0, [])
    return res

s = input().strip()
for ip in restoreIpAddresses(s):
    print(ip)
```

---

### 491. 非递减子序列

**M - 暴力解**
```
枚举所有子序列，检查是否非递减，用set去重
Time: O(2^n * n), Space: O(2^n * n)
```

**I - 边界分析**
- 上界：O(2^n)
- 目标：回溯+同层去重（不能排序，会破坏原序）

**K - 关键词触发**
- 子序列+非递减 → 当前值>=path末尾才可选
- 去重+不能排序 → 同层用set记录已选值

**E - 优化方案**
- 回溯+同层set去重：每层用set记录已选值，避免同层重复选择
- Time: 远小于O(2^n), Space: O(n)

**核心代码片段**
```python
def findSubsequences(nums):
    res = []
    def backtrack(start, path):
        if len(path) >= 2:
            res.append(path[:])
        used = set()  # 同层去重
        for i in range(start, len(nums)):
            if nums[i] in used: continue
            if path and nums[i] < path[-1]: continue
            used.add(nums[i])
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def findSubsequences(nums):
    res = []
    def backtrack(start, path):
        if len(path) >= 2:
            res.append(path[:])
        used = set()
        for i in range(start, len(nums)):
            if nums[i] in used: continue
            if path and nums[i] < path[-1]: continue
            used.add(nums[i])
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return res

nums = list(map(int, input().split()))
for seq in findSubsequences(nums):
    print(' '.join(map(str, seq)))
```

---

### 51. N皇后

**M - 暴力解**
```
枚举所有n^n种放置方案，检查是否合法
Time: O(n^n), Space: O(n^2)
```

**I - 边界分析**
- 上界：O(n^n)
- 下界：O(n!)（每行只能放一个）
- 目标：O(n!)

**K - 关键词触发**
- 棋盘+冲突检测 → 逐行放置，用集合记录列/对角线冲突
- 主对角线：row-col相同
- 副对角线：row+col相同

**E - 优化方案**
- 回溯+集合记录冲突：逐行放置，用3个set记录列/主对角线/副对角线
- Time: O(n!), Space: O(n^2)

**核心代码片段**
```python
def solveNQueens(n):
    res = []
    board = [['.'] * n for _ in range(n)]
    cols, diag1, diag2 = set(), set(), set()

    def backtrack(row):
        if row == n:
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    backtrack(0)
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solveNQueens(n):
    res = []
    board = [['.'] * n for _ in range(n)]
    cols, diag1, diag2 = set(), set(), set()

    def backtrack(row):
        if row == n:
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    backtrack(0)
    return res

n = int(input())
solutions = solveNQueens(n)
for sol in solutions:
    for row in sol:
        print(row)
    print()
```

---

### 37. 解数独

**M - 暴力解**
```
逐格尝试1-9，检查行/列/宫格合法性
Time: O(9^(n*n)), Space: O(n^2)
```

**I - 边界分析**
- 上界：O(9^81)
- 目标：回溯+剪枝大幅减少搜索

**K - 关键词触发**
- 棋盘+约束 → 逐格回溯，检查行/列/宫格
- 找到一个解即返回 → 回溯返回True/False

**E - 优化方案**
- 回溯+合法性检查：逐格尝试1-9，检查行/列/宫格，找到解返回True
- 优化：用集合预处理每行/列/宫格已有数字
- Time: 剪枝后远小于指数级, Space: O(1)

**核心代码片段**
```python
def solveSudoku(board):
    def isValid(row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_r, box_r + 3):
            for j in range(box_c, box_c + 3):
                if board[i][j] == num:
                    return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if isValid(i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    backtrack()
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solveSudoku(board):
    def isValid(row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_r, box_r + 3):
            for j in range(box_c, box_c + 3):
                if board[i][j] == num:
                    return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if isValid(i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    backtrack()

board = []
for _ in range(9):
    board.append(list(input().strip()))
solveSudoku(board)
for row in board:
    print(''.join(row))
```

---

### 394. 字符串解码 (Phase2测试)

**M - 暴力解**
```
递归解析：遇数字记录倍数，遇[递归，遇]返回
Time: O(n * maxK), Space: O(n)
```

**I - 边界分析**
- 上界：O(n * maxK)（maxK是最大重复次数）
- 目标：栈一次遍历

**K - 关键词触发**
- 括号匹配+嵌套 → 栈
- 遇[压入当前串和倍数，遇]弹出拼接

**E - 优化方案**
- 栈：遇[压入(当前串, 倍数)，遇]弹出并重复拼接
- Time: O(n * maxK), Space: O(n)

**核心代码片段**
```python
def decodeString(s):
    stack = []
    num = 0
    curr = ''
    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == '[':
            stack.append((curr, num))
            curr, num = '', 0
        elif c == ']':
            prev_str, k = stack.pop()
            curr = prev_str + curr * k
        else:
            curr += c
    return curr
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def decodeString(s):
    stack = []
    num = 0
    curr = ''
    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == '[':
            stack.append((curr, num))
            curr, num = '', 0
        elif c == ']':
            prev_str, k = stack.pop()
            curr = prev_str + curr * k
        else:
            curr += c
    return curr

s = input().strip()
print(decodeString(s))
```

---

### 103. 二叉树的锯齿形层序遍历 (Phase2测试)

**M - 暴力解**
```
BFS层序遍历，奇数层反转
Time: O(n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n)
- 已是最优

**K - 关键词触发**
- 层序遍历+变化 → BFS+层号判断
- 锯齿形 → 偶数层正序，奇数层反转

**E - 优化方案**
- BFS+层号标记：用level变量，奇数层反转当前层结果
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def zigzagLevelOrder(root):
    if not root: return []
    res, queue = [], [root]
    level = 0
    while queue:
        size = len(queue)
        curr_level = []
        for _ in range(size):
            node = queue.pop(0)
            curr_level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        if level % 2 == 1:
            curr_level.reverse()
        res.append(curr_level)
        level += 1
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

def buildTree(vals):
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

def zigzagLevelOrder(root):
    if not root: return []
    res, queue = [], [root]
    level = 0
    while queue:
        size = len(queue)
        curr_level = []
        for _ in range(size):
            node = queue.pop(0)
            curr_level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        if level % 2 == 1:
            curr_level.reverse()
        res.append(curr_level)
        level += 1
    return res

vals = input().strip().split()
root = buildTree(vals)
for level in zigzagLevelOrder(root):
    print(' '.join(map(str, level)))
```

---

### 450. 删除二叉搜索树中的节点 (Phase2测试)

**M - 暴力解**
```
递归查找+删除，分无子/单子/双子处理
Time: O(h), Space: O(h)
```

**I - 边界分析**
- 上界：O(n)（退化链表）
- 下界：O(log n)（平衡树）
- 目标：O(h)

**K - 关键词触发**
- BST+删除 → 递归+BST性质
- 双子节点 → 找右子树最小值（或左子树最大值）替换

**E - 优化方案**
- 递归：无子返回None，单子返回子节点，双子用右子树最小值替换
- Time: O(h), Space: O(h)

**核心代码片段**
```python
def deleteNode(root, key):
    if not root: return None
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        # 双子：找右子树最小值
        minNode = root.right
        while minNode.left:
            minNode = minNode.left
        root.val = minNode.val
        root.right = deleteNode(root.right, minNode.val)
    return root
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

def buildBST(vals):
    if not vals: return None
    root = None
    for val in vals:
        if val != 'null':
            root = insertBST(root, int(val))
    return root

def insertBST(root, val):
    if not root: return TreeNode(val)
    if val < root.val:
        root.left = insertBST(root.left, val)
    else:
        root.right = insertBST(root.right, val)
    return root

def deleteNode(root, key):
    if not root: return None
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        minNode = root.right
        while minNode.left:
            minNode = minNode.left
        root.val = minNode.val
        root.right = deleteNode(root.right, minNode.val)
    return root

def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

vals = input().strip().split()
key = int(input())
root = buildBST(vals)
root = deleteNode(root, key)
print(' '.join(map(str, inorder(root))))
```

---

### 22. 括号生成 (Phase2测试)

**M - 暴力解**
```
生成所有2^(2n)种括号组合，检查合法性
Time: O(2^(2n) * n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^(2n))
- 下界：O(C(2n,n)/(n+1))（卡特兰数）
- 目标：O(C(2n,n)/(n+1))

**K - 关键词触发**
- 括号+合法性 → 回溯，左括号剩余>0可加左，右>左可加右
- 剪枝 → 只生成合法序列

**E - 优化方案**
- 回溯+剪枝：left剩余>0加左，right>left加右
- Time: O(C(2n,n)/(n+1)), Space: O(n)

**核心代码片段**
```python
def generateParenthesis(n):
    res = []
    def backtrack(left, right, path):
        if left == 0 and right == 0:
            res.append(''.join(path))
            return
        if left > 0:
            path.append('(')
            backtrack(left - 1, right, path)
            path.pop()
        if right > left:
            path.append(')')
            backtrack(left, right - 1, path)
            path.pop()
    backtrack(n, n, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def generateParenthesis(n):
    res = []
    def backtrack(left, right, path):
        if left == 0 and right == 0:
            res.append(''.join(path))
            return
        if left > 0:
            path.append('(')
            backtrack(left - 1, right, path)
            path.pop()
        if right > left:
            path.append(')')
            backtrack(left, right - 1, path)
            path.pop()
    backtrack(n, n, [])
    return res

n = int(input())
for paren in generateParenthesis(n):
    print(paren)
```

---

### 79. 单词搜索 (Phase2测试)

**M - 暴力解**
```
从每个格子开始DFS，检查是否匹配单词
Time: O(m*n*4^L), Space: O(L)
```

**I - 边界分析**
- 上界：O(m*n*4^L)（L是单词长度）
- 目标：回溯+剪枝

**K - 关键词触发**
- 棋盘+路径搜索 → 回溯+DFS
- 不重复访问 → 标记visited或原地修改

**E - 优化方案**
- 回溯+DFS：从首字母位置四方向DFS，原地标记已访问
- Time: O(m*n*4^L), Space: O(L)

**核心代码片段**
```python
def exist(board, word):
    m, n = len(board), len(board[0])
    def dfs(i, j, k):
        if k == len(word): return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        found = (dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or
                 dfs(i,j+1,k+1) or dfs(i,j-1,k+1))
        board[i][j] = tmp
        return found

    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0] and dfs(i, j, 0):
                return True
    return False
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def exist(board, word):
    m, n = len(board), len(board[0])
    def dfs(i, j, k):
        if k == len(word): return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        found = (dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or
                 dfs(i,j+1,k+1) or dfs(i,j-1,k+1))
        board[i][j] = tmp
        return found

    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0] and dfs(i, j, 0):
                return True
    return False

m, n = map(int, input().split())
board = []
for _ in range(m):
    board.append(list(input().strip()))
word = input().strip()
print("true" if exist(board, word) else "false")
```

---
