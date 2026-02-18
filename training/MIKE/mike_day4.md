---

### Day 4 — 字符串

---

### 344. 反转字符串

**M - 暴力解**

```
For each pair (i, n-1-i) where i < n//2:
    swap s[i] and s[n-1-i]
Time: O(n), Space: O(1)
```

Already optimal — two-pointer swap is the naive and best approach.

**I - 边界分析**

- 上界：O(n) — must touch every element at least once
- 下界：O(n) — same reason
- 目标：O(n) time, O(1) space (in-place)

**K - 关键词触发**

- "原地修改" → 双指针 swap
- "反转" → 首尾对称交换

**E - 优化方案**

双指针从两端向中间逼近，逐对交换。Time O(n), Space O(1)。

**核心代码片段**

```python
def reverseString(s):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def reverseString(s):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1

s = list(input().strip())
reverseString(s)
print(''.join(s))
```

---

### 541. 反转字符串 II

**M - 暴力解**

```
For i in range(0, n, 2k):
    reverse s[i : i+k]
    keep s[i+k : i+2k] as is
If remaining < k, reverse all remaining
Time: O(n), Space: O(n) (string immutable in Python)
```

**I - 边界分析**

- 上界：O(n) — single pass with step 2k
- 下界：O(n) — must read every character
- 目标：O(n) time
- 边界：剩余不足 k 个时全部反转；剩余 >= k 但 < 2k 时只反转前 k 个

**K - 关键词触发**

- "每隔2k" → 步长为 2k 的分段处理
- "反转前k个" → 切片反转

**E - 优化方案**

转为 list，按 2k 步长遍历，对每段前 k 个字符原地反转。Time O(n), Space O(n)。

**核心代码片段**

```python
def reverseStr(s, k):
    a = list(s)
    for i in range(0, len(a), 2 * k):
        a[i:i+k] = a[i:i+k][::-1]
    return ''.join(a)
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def reverseStr(s, k):
    a = list(s)
    for i in range(0, len(a), 2 * k):
        a[i:i+k] = a[i:i+k][::-1]
    return ''.join(a)

s = input().strip()
k = int(input().strip())
print(reverseStr(s, k))
```

---

### 151. 反转字符串中的单词

**M - 暴力解**

```
Split string by spaces (filter empty)
Reverse the list of words
Join with single space
Time: O(n), Space: O(n)
```

**I - 边界分析**

- 上界：O(n)
- 下界：O(n)
- 目标：O(n) time, O(n) space（Python 字符串不可变）
- 边界：前导/尾随空格、连续多个空格、单个单词

**K - 关键词触发**

- "反转单词顺序" → split + reverse
- "去除多余空格" → split() 自动处理

**E - 优化方案**

Python 的 `split()` 无参数时自动按任意空白分割并去除空串，然后 `[::-1]` 反转再 join。Time O(n), Space O(n)。

进阶（C++原地做法思路）：先整体反转字符串，再逐个单词反转，最后去多余空格。

**核心代码片段**

```python
def reverseWords(s):
    return ' '.join(s.split()[::-1])
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def reverseWords(s):
    return ' '.join(s.split()[::-1])

s = input().strip()
print(reverseWords(s))
```

---

### 28. 找出字符串中第一个匹配项的下标

**M - 暴力解**

```
For each position i in haystack:
    check if haystack[i:i+m] == needle
    if match, return i
Return -1
Time: O(n*m), Space: O(1)
```

**I - 边界分析**

- 上界：O(n*m) — brute force
- 下界：O(n+m) — KMP / Z-algorithm
- 目标：O(n+m) time
- 边界：needle 为空返回 0；needle 比 haystack 长返回 -1

**K - 关键词触发**

- "字符串匹配" → KMP 算法
- "模式串" → 构建 next/前缀表

**E - 优化方案**

KMP 算法：先构建 next 数组（最长相等前后缀），匹配失败时利用 next 数组跳过已匹配部分。Time O(n+m), Space O(m)。

**核心代码片段**

```python
def strStr(haystack, needle):
    n, m = len(haystack), len(needle)
    if m == 0:
        return 0
    # 构建 next 数组（前缀表，不减一版本）
    nxt = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and needle[i] != needle[j]:
            j = nxt[j - 1]
        if needle[i] == needle[j]:
            j += 1
        nxt[i] = j
    # KMP 匹配
    j = 0
    for i in range(n):
        while j > 0 and haystack[i] != needle[j]:
            j = nxt[j - 1]
        if haystack[i] == needle[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def strStr(haystack, needle):
    n, m = len(haystack), len(needle)
    if m == 0:
        return 0
    nxt = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and needle[i] != needle[j]:
            j = nxt[j - 1]
        if needle[i] == needle[j]:
            j += 1
        nxt[i] = j
    j = 0
    for i in range(n):
        while j > 0 and haystack[i] != needle[j]:
            j = nxt[j - 1]
        if haystack[i] == needle[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1

haystack = input().strip()
needle = input().strip()
print(strStr(haystack, needle))
```

---

### 459. 重复的子字符串

**M - 暴力解**

```
For length d from 1 to n//2:
    if n % d == 0:
        if s == s[:d] * (n // d):
            return True
Return False
Time: O(n * sqrt(n)) or O(n^2) worst case, Space: O(n)
```

**I - 边界分析**

- 上界：O(n^2) — brute force
- 下界：O(n) — KMP-based
- 目标：O(n) time
- 边界：长度为 1 的字符串返回 False；全相同字符返回 True

**K - 关键词触发**

- "重复子串" → KMP next 数组性质
- "周期" → 字符串周期 = n - next[n-1]，若 n % 周期 == 0 则由该子串重复构成

**E - 优化方案**

方法一（KMP）：构建 next 数组，若 `next[n-1] > 0` 且 `n % (n - next[n-1]) == 0`，则可由长度为 `n - next[n-1]` 的子串重复构成。Time O(n), Space O(n)。

方法二（拼接法）：`(s + s)[1:-1]` 中若包含 s，则 s 由重复子串构成。Time O(n), Space O(n)。

**核心代码片段**

```python
# 方法一：KMP
def repeatedSubstringPattern(s):
    n = len(s)
    nxt = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = nxt[j - 1]
        if s[i] == s[j]:
            j += 1
        nxt[i] = j
    period = n - nxt[n - 1]
    return nxt[n - 1] > 0 and n % period == 0

# 方法二：拼接法
def repeatedSubstringPattern2(s):
    return s in (s + s)[1:-1]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def repeatedSubstringPattern(s):
    n = len(s)
    nxt = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = nxt[j - 1]
        if s[i] == s[j]:
            j += 1
        nxt[i] = j
    period = n - nxt[n - 1]
    return nxt[n - 1] > 0 and n % period == 0

s = input().strip()
print("true" if repeatedSubstringPattern(s) else "false")
```

---

### 3. 无重复字符的最长子串

**M - 暴力解**

```
For each pair (i, j):
    check if s[i:j] has all unique chars
    track max length
Time: O(n^3), Space: O(min(n, charset))
```

**I - 边界分析**

- 上界：O(n^3) — brute force
- 下界：O(n) — sliding window
- 目标：O(n) time
- 边界：空串返回 0；全相同字符返回 1；全不同字符返回 n

**K - 关键词触发**

- "最长子串" + "无重复" → 滑动窗口
- "子串"（连续）→ 双指针维护窗口

**E - 优化方案**

滑动窗口 + 哈希表记录字符最近出现位置。右指针扩展，遇到重复字符时左指针跳到重复位置 + 1。Time O(n), Space O(min(n, charset))。

**核心代码片段**

```python
def lengthOfLongestSubstring(s):
    last = {}  # char -> last index
    ans = 0
    left = 0
    for right, c in enumerate(s):
        if c in last and last[c] >= left:
            left = last[c] + 1
        last[c] = right
        ans = max(ans, right - left + 1)
    return ans
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def lengthOfLongestSubstring(s):
    last = {}
    ans = 0
    left = 0
    for right, c in enumerate(s):
        if c in last and last[c] >= left:
            left = last[c] + 1
        last[c] = right
        ans = max(ans, right - left + 1)
    return ans

s = input().strip()
print(lengthOfLongestSubstring(s))
```

---

### 5. 最长回文子串

**M - 暴力解**

```
For each pair (i, j):
    check if s[i:j+1] is palindrome
    track longest
Time: O(n^3), Space: O(1)
```

**I - 边界分析**

- 上界：O(n^3) — brute force
- 下界：O(n) — Manacher 算法
- 目标：O(n^2) 中心扩展（面试常考），O(n) Manacher（进阶）
- 边界：长度为 1 返回自身；全相同字符返回整个串

**K - 关键词触发**

- "回文" → 中心扩展 / DP / Manacher
- "最长" → 枚举中心，向两侧扩展

**E - 优化方案**

中心扩展法：枚举每个位置作为中心（奇数长度）和每对相邻位置作为中心（偶数长度），向两侧扩展。Time O(n^2), Space O(1)。

**核心代码片段**

```python
def longestPalindrome(s):
    n = len(s)
    start, maxlen = 0, 1

    def expand(l, r):
        nonlocal start, maxlen
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > maxlen:
                start = l
                maxlen = r - l + 1
            l -= 1
            r += 1

    for i in range(n):
        expand(i, i)      # 奇数长度
        expand(i, i + 1)  # 偶数长度
    return s[start:start + maxlen]
```

**ACM模式完整代码**

```python
import sys
input = sys.stdin.readline

def longestPalindrome(s):
    n = len(s)
    if n < 2:
        return s
    start, maxlen = 0, 1

    def expand(l, r):
        nonlocal start, maxlen
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > maxlen:
                start = l
                maxlen = r - l + 1
            l -= 1
            r += 1

    for i in range(n):
        expand(i, i)
        expand(i, i + 1)
    return s[start:start + maxlen]

s = input().strip()
print(longestPalindrome(s))
```

---

### 438. 找到字符串中所有字母异位词

**M - 暴力解**

```
For each window of length len(p) in s:
    sort the window and compare with sorted p
    if equal, record start index
Time: O(n * m * log(m)), Space: O(m)
```

**I - 边界分析**

- 上界：O(n * m * log(m)) — sort each window
- 下界：O(n) — sliding window with frequency count
- 目标：O(n) time（n = len(s)）
- 边界：len(p) > len(s) 返回空列表；p 和 s 均只含小写字母

**K - 关键词触发**

- "异位词" → 字符频率相同
- "所有子串" + 固定长度 → 定长滑动窗口
- "字母异位词" → 26 位频率数组 / Counter

**E - 优化方案**

定长滑动窗口 + 频率数组差值计数。维护窗口内与 p 的字符频率差异数 `diff`，当 `diff == 0` 时记录起始位置。Time O(n), Space O(26) = O(1)。

**核心代码片段**

```python
from collections import Counter

def findAnagrams(s, p):
    n, m = len(s), len(p)
    if m > n:
        return []
    ans = []
    p_cnt = Counter(p)
    w_cnt = Counter(s[:m])
    if w_cnt == p_cnt:
        ans.append(0)
    for i in range(m, n):
        w_cnt[s[i]] += 1
        left_char = s[i - m]
        w_cnt[left_char] -= 1
        if w_cnt[left_char] == 0:
            del w_cnt[left_char]
        if w_cnt == p_cnt:
            ans.append(i - m + 1)
    return ans
```

**ACM模式完整代码**

```python
import sys
from collections import Counter
input = sys.stdin.readline

def findAnagrams(s, p):
    n, m = len(s), len(p)
    if m > n:
        return []
    ans = []
    p_cnt = Counter(p)
    w_cnt = Counter(s[:m])
    if w_cnt == p_cnt:
        ans.append(0)
    for i in range(m, n):
        w_cnt[s[i]] += 1
        left_char = s[i - m]
        w_cnt[left_char] -= 1
        if w_cnt[left_char] == 0:
            del w_cnt[left_char]
        if w_cnt == p_cnt:
            ans.append(i - m + 1)
    return ans

s = input().strip()
p = input().strip()
result = findAnagrams(s, p)
print(result if result else "[]")
```
