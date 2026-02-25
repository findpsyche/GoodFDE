数组
=
977有序数组平方
![[Pasted image 20260224155500.png]]


数组方法：
数组 双指针 （二分查找 快慢指针  滑动窗口（从两个for 暴力推动 转换为1个for ）  三指针 （两个搜 一个输出）

滑动窗口： 毛毛虫向前移动
## 🔍 分解理解
### 1. 条件表达式的结构
```plaintext
返回值1 if 条件 else 返回值2
```
- 如果条件为真，返回 "返回值 1"
- 如果条件为假，返回 "返回值 2"
### 2. 这行代码的具体含义

```python
return min_len if min_len != float('inf') else 0
```
可以拆分为：
```python
if min_len != float('inf'):
    return min_len  # 如果找到了满足条件的子数组
else:
    return 0       # 如果没有找到满足条件的子数组
```
### 3. 为什么这样写？
- `min_len`初始化为`float('inf')`（无穷大）
- 如果找到了满足条件的子数组，`min_len`会被更新为一个具体的数字
- 如果没有找到任何满足条件的子数组，`min_len`仍然是`float('inf')`
- 所以需要检查`min_len`是否还是无穷大，来决定返回什么值



---

链表
=
数据定义:
```python
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
```
