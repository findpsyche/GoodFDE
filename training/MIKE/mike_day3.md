---

### Day 3 — 哈希表

---

### 242. 有效的字母异位词

**M - 暴力解**
```
Sort both strings, compare
Time: O(n log n), Space: O(n)
```

**I - 边界分析**
- 上界：O(n log n) 排序比较
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(1) space（固定26字母）

**K - 关键词触发**
- 字母频次统计 → 哈希表/数组计数
- 固定字符集(26小写) → 长度26数组替代哈希表

**E - 优化方案**
- 用长度26的数组统计字符频次，s中+1，t中-1，最终全为0则是异位词
- Time: O(n), Space: O(1)

**核心代码片段**
```python
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    cnt = [0] * 26
    for a, b in zip(s, t):
        cnt[ord(a) - ord('a')] += 1
        cnt[ord(b) - ord('a')] -= 1
    return all(x == 0 for x in cnt)
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    s = input().strip()
    t = input().strip()
    if len(s) != len(t):
        print("false")
        return
    cnt = [0] * 26
    for a, b in zip(s, t):
        cnt[ord(a) - ord('a')] += 1
        cnt[ord(b) - ord('a')] -= 1
    print("true" if all(x == 0 for x in cnt) else "false")

solve()
```

---

### 349. 两个数组的交集

**M - 暴力解**
```
For each element in nums1, scan nums2 to check existence
Deduplicate result
Time: O(m * n), Space: O(min(m, n))
```

**I - 边界分析**
- 上界：O(m * n) 暴力双循环
- 下界：O(m + n) 至少遍历两数组
- 目标：O(m + n) time, O(m) space

**K - 关键词触发**
- 查找元素是否存在 → 哈希集合 set
- 去重 → set天然去重

**E - 优化方案**
- 将nums1转为set，遍历nums2检查是否在set中，用set收集结果自动去重
- Time: O(m + n), Space: O(m)

**核心代码片段**
```python
def intersection(nums1: list, nums2: list) -> list:
    set1 = set(nums1)
    return list(set1 & set(nums2))
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums1 = list(map(int, input().split()))
    nums2 = list(map(int, input().split()))
    res = set(nums1) & set(nums2)
    print(" ".join(map(str, sorted(res))))

solve()
```

---

### 202. 快乐数

**M - 暴力解**
```
Repeatedly compute sum of squares of digits
If seen before (cycle detected), return false
If reaches 1, return true
Time: O(?) hard to bound, Space: O(k) for visited set
```

**I - 边界分析**
- 对于int范围，各位平方和最大为 9^2 * 10 = 810（10位数），所以数值会迅速收敛到 < 810
- 循环检测最多几百步
- 目标：O(log n) per step, O(k) space 或 O(1) space用快慢指针

**K - 关键词触发**
- 循环检测 → 哈希集合 / 快慢指针（Floyd判圈）
- 各位数字操作 → n % 10, n // 10

**E - 优化方案**
- 方案1：哈希集合记录出现过的数，检测循环。Time: O(k * log n), Space: O(k)
- 方案2：快慢指针，Space: O(1)

**核心代码片段**
```python
def isHappy(n: int) -> bool:
    def get_next(num):
        s = 0
        while num:
            num, d = divmod(num, 10)
            s += d * d
        return s

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)
    return n == 1
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())

    def get_next(num):
        s = 0
        while num:
            num, d = divmod(num, 10)
            s += d * d
        return s

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)
    print("true" if n == 1 else "false")

solve()
```

---

### 1. 两数之和

**M - 暴力解**
```
For each pair (i, j) where i < j, check if nums[i] + nums[j] == target
Time: O(n^2), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^2) 枚举所有对
- 下界：O(n) 至少遍历一次
- 目标：O(n) time, O(n) space
- 题目保证恰好一个解

**K - 关键词触发**
- 两数之和 = target → 哈希表存 complement
- 需要返回下标 → 不能排序（会丢失原始下标），必须用哈希表

**E - 优化方案**
- 一次遍历，哈希表存 {值: 下标}，对每个num查找 target - num 是否已在表中
- Time: O(n), Space: O(n)

**核心代码片段**
```python
def twoSum(nums: list, target: int) -> list:
    d = {}
    for i, x in enumerate(nums):
        comp = target - x
        if comp in d:
            return [d[comp], i]
        d[x] = i
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    target = int(input())
    d = {}
    for i, x in enumerate(nums):
        comp = target - x
        if comp in d:
            print(d[comp], i)
            return
        d[x] = i

solve()
```

---

### 454. 四数相加 II

**M - 暴力解**
```
Four nested loops, check if a + b + c + d == 0
Time: O(n^4), Space: O(1)
```

**I - 边界分析**
- 上界：O(n^4) 四重循环
- 下界：O(n^2) 至少枚举两两组合
- 目标：O(n^2) time, O(n^2) space
- n <= 200, n^2 = 40000 完全可行

**K - 关键词触发**
- 四数之和为0 → 分组：(A+B) + (C+D) = 0 → 转化为两数之和
- 不要求去重/返回下标 → 只需计数 → 哈希表计数

**E - 优化方案**
- 枚举A+B所有和存入哈希表{sum: count}，再枚举C+D查找 -(c+d) 的count
- Time: O(n^2), Space: O(n^2)

**核心代码片段**
```python
from collections import Counter

def fourSumCount(nums1, nums2, nums3, nums4) -> int:
    ab = Counter(a + b for a in nums1 for b in nums2)
    return sum(ab[-(c + d)] for c in nums3 for d in nums4)
```

**ACM模式完整代码**
```python
import sys
from collections import Counter
input = sys.stdin.readline

def solve():
    n = int(input())
    nums1 = list(map(int, input().split()))
    nums2 = list(map(int, input().split()))
    nums3 = list(map(int, input().split()))
    nums4 = list(map(int, input().split()))

    ab = Counter(a + b for a in nums1 for b in nums2)
    res = sum(ab[-(c + d)] for c in nums3 for d in nums4)
    print(res)

solve()
```

---

### 15. 三数之和

**M - 暴力解**
```
Three nested loops, find all triplets summing to 0
Skip duplicates by sorting and checking
Time: O(n^3), Space: O(1) excluding output
```

**I - 边界分析**
- 上界：O(n^3) 三重循环
- 下界：O(n^2) 固定一个数后双指针
- 目标：O(n^2) time, O(1) space（排序原地）
- 关键难点：去重逻辑

**K - 关键词触发**
- 三数之和 → 排序 + 固定一个 + 双指针
- 不重复三元组 → 排序后跳过相同元素去重
- 已排序数组两数之和 → 双指针

**E - 优化方案**
- 排序数组，枚举第一个数i，对剩余部分用双指针(left, right)找两数之和 = -nums[i]
- 去重：i跳过相同，left/right找到解后跳过相同
- Time: O(n^2), Space: O(1)

**核心代码片段**
```python
def threeSum(nums: list) -> list:
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1
    for triplet in res:
        print(" ".join(map(str, triplet)))

solve()
```

---

### 18. 四数之和

**M - 暴力解**
```
Four nested loops, find all quadruplets summing to target
Time: O(n^4), Space: O(1) excluding output
```

**I - 边界分析**
- 上界：O(n^4) 四重循环
- 下界：O(n^3) 固定两个数后双指针
- 目标：O(n^3) time, O(1) space
- 关键难点：两层去重 + 剪枝

**K - 关键词触发**
- 四数之和 → 排序 + 固定两个 + 双指针（三数之和的扩展）
- 不重复四元组 → 每层循环跳过重复元素
- 剪枝 → 最小和 > target 提前break，最大和 < target 提前continue

**E - 优化方案**
- 排序，两层循环固定前两个数(i, j)，内层双指针(l, r)
- 每层去重 + 剪枝优化
- Time: O(n^3), Space: O(1)

**核心代码片段**
```python
def fourSum(nums: list, target: int) -> list:
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
            break
        if nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                break
            if nums[i] + nums[j] + nums[n-2] + nums[n-1] < target:
                continue
            l, r = j + 1, n - 1
            while l < r:
                s = nums[i] + nums[j] + nums[l] + nums[r]
                if s < target:
                    l += 1
                elif s > target:
                    r -= 1
                else:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
    return res
```

**ACM模式完整代码**
```python
import sys
input = sys.stdin.readline

def solve():
    nums = list(map(int, input().split()))
    target = int(input())
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
            break
        if nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                break
            if nums[i] + nums[j] + nums[n-2] + nums[n-1] < target:
                continue
            l, r = j + 1, n - 1
            while l < r:
                s = nums[i] + nums[j] + nums[l] + nums[r]
                if s < target:
                    l += 1
                elif s > target:
                    r -= 1
                else:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
    for quad in res:
        print(" ".join(map(str, quad)))

solve()
```

---

### 49. 字母异位词分组

**M - 暴力解**
```
For each pair of strings, check if they are anagrams (sort and compare)
Group by equivalence classes
Time: O(n^2 * k log k) where k = max string length, Space: O(n * k)
```

**I - 边界分析**
- 上界：O(n^2 * k log k) 两两比较
- 下界：O(n * k) 至少遍历所有字符
- 目标：O(n * k log k) 或 O(n * k) time

**K - 关键词触发**
- 分组 → 哈希表，key = 规范化表示
- 字母异位词 → 排序后相同 / 字符频次相同
- 规范化key → sorted string 或 tuple(count)

**E - 优化方案**
- 方案1：排序作key，将每个字符串排序后作为哈希表的key。Time: O(n * k log k)
- 方案2：计数作key，用26位字符频次元组作key。Time: O(n * k)，常数稍大但渐进更优

**核心代码片段**
```python
from collections import defaultdict

def groupAnagrams(strs: list) -> list:
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

**ACM模式完整代码**
```python
import sys
from collections import defaultdict
input = sys.stdin.readline

def solve():
    strs = input().split()
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    for group in groups.values():
        print(" ".join(group))

solve()
```
