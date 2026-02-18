---

### Day 9 — 回溯算法

---

### 77. 组合

**M - 暴力解**
```
多重循环枚举所有k个数的组合
Time: O(n^k), Space: O(k)
```

**I - 边界分析**
- 上界：O(n^k)
- 下界：O(C(n,k) * k)（组合数量×每个组合长度）
- 目标：O(C(n,k) * k)

**K - 关键词触发**
- 组合 → 回溯，startIndex保证不重复
- 剪枝 → 剩余元素不够k个时提前返回

**E - 优化方案**
- 回溯+剪枝：从startIndex开始选择，剩余不够则剪枝
- Time: O(C(n,k) * k), Space: O(k)

**核心代码片段**
```python
def combine(n, k):
    res = []
    def backtrack(start, path):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n - (k - len(path)) + 2):  # 剪枝
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def combine(n, k):
    res = []
    def backtrack(start, path):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, [])
    return res

n, k = map(int, input().split())
for combo in combine(n, k):
    print(' '.join(map(str, combo)))
```

---

### 216. 组合总和 III

**M - 暴力解**
```
枚举1-9中所有k个数的组合，检查和是否为n
Time: O(C(9,k) * k), Space: O(k)
```

**I - 边界分析**
- 上界：O(C(9,k))
- 目标：通过剪枝大幅减少搜索空间

**K - 关键词触发**
- 组合+和约束 → 回溯+剪枝（和超过n提前返回）

**E - 优化方案**
- 回溯+双重剪枝：和超过n剪枝 + 剩余元素不够剪枝
- Time: O(C(9,k)), Space: O(k)

**核心代码片段**
```python
def combinationSum3(k, n):
    res = []
    def backtrack(start, path, remain):
        if len(path) == k:
            if remain == 0:
                res.append(path[:])
            return
        for i in range(start, 10):
            if i > remain: break  # 剪枝
            path.append(i)
            backtrack(i + 1, path, remain - i)
            path.pop()
    backtrack(1, [], n)
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def combinationSum3(k, n):
    res = []
    def backtrack(start, path, remain):
        if len(path) == k:
            if remain == 0:
                res.append(path[:])
            return
        for i in range(start, 10):
            if i > remain: break
            path.append(i)
            backtrack(i + 1, path, remain - i)
            path.pop()
    backtrack(1, [], n)
    return res

k, n = map(int, input().split())
for combo in combinationSum3(k, n):
    print(' '.join(map(str, combo)))
```

---

### 17. 电话号码的字母组合

**M - 暴力解**
```
多重循环，每位数字对应的字母逐一组合
Time: O(4^n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(4^n * n)（每个组合长度n）
- 已是最优（必须枚举所有组合）

**K - 关键词触发**
- 多选一组合 → 回溯，每层遍历当前数字对应的字母

**E - 优化方案**
- 回溯：建立数字→字母映射，逐位选择
- Time: O(4^n * n), Space: O(n)

**核心代码片段**
```python
def letterCombinations(digits):
    if not digits: return []
    phone = {'2':'abc','3':'def','4':'ghi','5':'jkl',
             '6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    res = []
    def backtrack(idx, path):
        if idx == len(digits):
            res.append(''.join(path))
            return
        for c in phone[digits[idx]]:
            path.append(c)
            backtrack(idx + 1, path)
            path.pop()
    backtrack(0, [])
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def letterCombinations(digits):
    if not digits: return []
    phone = {'2':'abc','3':'def','4':'ghi','5':'jkl',
             '6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    res = []
    def backtrack(idx, path):
        if idx == len(digits):
            res.append(''.join(path))
            return
        for c in phone[digits[idx]]:
            path.append(c)
            backtrack(idx + 1, path)
            path.pop()
    backtrack(0, [])
    return res

digits = input().strip()
res = letterCombinations(digits)
print(' '.join(res) if res else "")
```

---

### 39. 组合总和

**M - 暴力解**
```
枚举所有可能的组合（允许重复选择），检查和
Time: 指数级, Space: O(target/min)
```

**I - 边界分析**
- 上界：指数级
- 目标：排序+剪枝大幅减少搜索

**K - 关键词触发**
- 组合+可重复选择 → 回溯，startIndex不+1（允许重复选自己）
- 排序+剪枝 → 当前值>剩余则break

**E - 优化方案**
- 排序后回溯+剪枝：从当前位置开始（允许重复），超过target则break
- Time: 剪枝后远小于指数级, Space: O(target/min)

**核心代码片段**
```python
def combinationSum(candidates, target):
    candidates.sort()
    res = []
    def backtrack(start, path, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break
            path.append(candidates[i])
            backtrack(i, path, remain - candidates[i])  # i不+1，允许重复
            path.pop()
    backtrack(0, [], target)
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def combinationSum(candidates, target):
    candidates.sort()
    res = []
    def backtrack(start, path, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break
            path.append(candidates[i])
            backtrack(i, path, remain - candidates[i])
            path.pop()
    backtrack(0, [], target)
    return res

candidates = list(map(int, input().split()))
target = int(input())
for combo in combinationSum(candidates, target):
    print(' '.join(map(str, combo)))
```

---

### 40. 组合总和 II

**M - 暴力解**
```
枚举所有组合（每个元素只用一次），用set去重
Time: O(2^n), Space: O(n)
```

**I - 边界分析**
- 上界：O(2^n)
- 目标：排序+同层去重避免重复组合

**K - 关键词触发**
- 含重复元素+不重复组合 → 排序+同层去重（i>start且nums[i]==nums[i-1]则跳过）

**E - 优化方案**
- 排序+回溯+同层去重+剪枝
- Time: 远小于O(2^n), Space: O(n)

**核心代码片段**
```python
def combinationSum2(candidates, target):
    candidates.sort()
    res = []
    def backtrack(start, path, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break
            if i > start and candidates[i] == candidates[i - 1]: continue  # 同层去重
            path.append(candidates[i])
            backtrack(i + 1, path, remain - candidates[i])
            path.pop()
    backtrack(0, [], target)
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def combinationSum2(candidates, target):
    candidates.sort()
    res = []
    def backtrack(start, path, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break
            if i > start and candidates[i] == candidates[i - 1]: continue
            path.append(candidates[i])
            backtrack(i + 1, path, remain - candidates[i])
            path.pop()
    backtrack(0, [], target)
    return res

candidates = list(map(int, input().split()))
target = int(input())
for combo in combinationSum2(candidates, target):
    print(' '.join(map(str, combo)))
```

---

### 46. 全排列

**M - 暴力解**
```
回溯+used数组标记已使用元素
Time: O(n! * n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(n! * n)
- 已是最优

**K - 关键词触发**
- 排列 → 回溯，每次从头遍历+used数组跳过已选
- 区别于组合：排列每次从0开始，组合从startIndex开始

**E - 优化方案**
- 交换法：原地交换省去used数组
- Time: O(n! * n), Space: O(n)

**核心代码片段**
```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return res

nums = list(map(int, input().split()))
for perm in permute(nums):
    print(' '.join(map(str, perm)))
```

---

### 47. 全排列 II

**M - 暴力解**
```
全排列后用set去重
Time: O(n! * n), Space: O(n! * n)
```

**I - 边界分析**
- 上界：O(n! * n)
- 目标：排序+同层去重避免生成重复排列

**K - 关键词触发**
- 含重复元素+不重复排列 → 排序+同层去重
- 去重条件：i>0且nums[i]==nums[i-1]且used[i-1]==False

**E - 优化方案**
- 排序+回溯+去重：同层相同元素且前一个未使用则跳过
- Time: 远小于O(n!), Space: O(n)

**核心代码片段**
```python
def permuteUnique(nums):
    nums.sort()
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def permuteUnique(nums):
    nums.sort()
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return res

nums = list(map(int, input().split()))
for perm in permuteUnique(nums):
    print(' '.join(map(str, perm)))
```

---

### 78. 子集

**M - 暴力解**
```
位运算枚举：n个元素有2^n个子集，用二进制位表示选/不选
Time: O(2^n * n), Space: O(n)
```

**I - 边界分析**
- 上界 = 下界 = O(2^n * n)
- 已是最优

**K - 关键词触发**
- 子集/幂集 → 回溯，每个节点都收集结果（不只是叶子）
- 区别于组合：子集在每层都收集，组合只在满足条件时收集

**E - 优化方案**
- 回溯：每次递归先将当前path加入结果，从startIndex开始遍历
- Time: O(2^n * n), Space: O(n)

**核心代码片段**
```python
def subsets(nums):
    res = []
    def backtrack(start, path):
        res.append(path[:])  # 每个节点都收集
        for i in range(start, len(nums)):
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

def subsets(nums):
    res = []
    def backtrack(start, path):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return res

nums = list(map(int, input().split()))
for sub in subsets(nums):
    print(' '.join(map(str, sub)) if sub else "(empty)")
```
