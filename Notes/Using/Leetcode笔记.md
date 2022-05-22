# Leetcode刷题笔记（python）

## 基础

我们从算法所占用的「时间」和「空间」两个维度考察算法之间的优劣。

- 时间维度：是指执行当前算法所消耗的时间，我们通常用「时间复杂度」来描述。
- 空间维度：是指执行当前算法需要占用多少内存空间，我们通常用「空间复杂度」来描述。

我们通常使用「 **大O符号表示法** 」描述时间的复杂度，该符号又称为**渐进符号**。

用渐进符号可以将复杂度分为常数阶O(1)、线性阶O(n)、指数阶O(2^n)、对数阶O(logn)、线性对数阶O(nlogn)

复杂度排名：

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115713666.png" alt="image-20210410115713666" style="zoom:50%;" />

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115804432.png" alt="image-20210410115804432" style="zoom:50%;" />

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115822169.png" alt="image-20210410115822169" style="zoom:50%;" />



## 内置函数

**set()**

当需要对一个列表进行去重操作的时候，可以使用`set()`函数。

```python3
obj = ['a','b','c','b','a']
print(set(obj))
# 输出：{'b', 'c', 'a'}
```

`set([iterable])`用于创建一个集合，集合里的元素是**无序且不重复的**。

集合对象创建后，还能使用**并集、交集、差集**功能。

```python3
A = set('hello')
B = set('world')

A.union(B) # 并集，输出：{'d', 'e', 'h', 'l', 'o', 'r', 'w'}
A.intersection(B) # 交集，输出：{'l', 'o'}
A.difference(B) # 差集，输出：{'d', 'r', 'w'}
```



**sorted()**

`sorted()`可以对任何可迭代对象进行排序，并返回列表。

注意：sorted(iterable)可作用于任意的对象，而sort是列表的方法，只能作用于列表，并且sorted是静态，无需使用对象调用

对列表升序操作：

```python3
a = sorted([2,4,3,7,1,9])
print(a)
# 输出：[1, 2, 3, 4, 7, 9]
```

对元组倒序操作：

```python3
sorted((4,1,9,6),reverse=True)
print(a)
# 输出：[9, 6, 4, 1]
```

使用参数：key，根据自定义规则，按字符串长度来排序：

```python3
chars = ['apple','watermelon','pear','banana']
a = sorted(chars,key=lambda x:len(x))
print(a)
# 输出：['pear', 'apple', 'banana', 'watermelon']
```

使用参数：key，根据自定义规则，对元组构成的列表进行排序：

```python3
tuple_list = [('A', 1,5), ('B', 3,2), ('C', 2,6)]
# key=lambda x: x[1]中可以任意选定x中可选的位置进行排序
a = sorted(tuple_list, key=lambda x: x[1])
print(a)
# 输出：[('A', 1, 5), ('C', 2, 6), ('B', 3, 2)]
```



**reversed()**

`reversed()`接受一个序列，将序列里的元素反转，并最终返回迭代器。

```python3
a = reversed('abcde')
print(list(a))
# 输出：['e', 'd', 'c', 'b', 'a']

b = reversed([2,3,4,5])
print(list(b))
# 输出：[5, 4, 3, 2]
```



**map()**

语法：`map(function, iterable, ...)`

**map() 会根据提供的函数对指定序列做映射**。第一个参数 function 以参数序列中的**每一个元素**调用 function 函数，第二个参数iterable为需要操作的参数序列。方法将返回包含每次 function 函数返回值的新列表。

做文本处理的时候，假如要对序列里的每个单词进行大写转化操作。

```python3
chars = ['apple','watermelon','pear','banana']
a = map(lambda x:x.upper(),chars)
print(list(a))
# 输出：['APPLE', 'WATERMELON', 'PEAR', 'BANANA']
```

举个例子，对列表里的每个数字作平方处理：

```python3
nums = [1,2,3,4]
a = map(lambda x:x*x,nums)
print(list(a))
# 输出：[1, 4, 9, 16]
```



**filter()**

语法：`filter(function, iterable)`

filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。

该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

找到列表中的所有奇数。

```python3
nums = [1,2,3,4,5,6]
a = filter(lambda x:x%2!=0,nums)
print(list(a))
# 输出：[1,3,5]
```

从许多单词里挑出包含字母`w`的单词。

```python3
chars = chars = ['apple','watermelon','pear','banana']
a = filter(lambda x:'w' in x,chars)
print(list(a))
# 输出：['watermelon']
```



**enumerate()**

这样一个场景，同时打印出序列里每一个元素和它对应的顺序号，我们用`enumerate()`函数做做看。

```python3
chars = ['apple','watermelon','pear','banana']
for i,j in enumerate(chars):
    print(i,j)

'''
输出：
0 apple
1 watermelon
2 pear
3 banana
'''
```

`enumerate`翻译过来是枚举、列举的意思，所以说`enumerate()`函数用于对序列里的元素进行顺序标注，返回(元素、索引)组成的迭代器。

再举个例子说明，对字符串进行标注，返回每个字母和其索引。

```python3
a = enumerate('abcd')
print(list(a))
# 输出：[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]
```



## 内置容器

#### **列表**

序列是 Python 中最基本的数据结构。序列中的每个值都有对应的位置值，称之为索引，用索引的方式可以访问列表中的元素。

特点：查询快、添加慢

**语法**

创建：

`list()`	创建列表（也可以用`l=[]`）

属性：

`len()`	返回列表元素的个数

`max()`	返回列表元素的最大值

`min()`	返回列表元素的最小值

`list(*params)`	将元组转换为列表	

操作：

| 函数                             | 描述                                                         |
| :------------------------------- | :----------------------------------------------------------- |
| l.append(obj)                  | 添加元素                                                     |
| l.remove(obj)                  | 移除列表中某个值的第一个匹配项                               |
| l.index(obj)                    | 从列表中找出某个值第一个匹配项的索引位置                     |
| l.clear()                     | 清空列表                                                     |
| l.reverse()                   | 反向列表元素                                                 |
| l.sort(key=None,reverse=False) | 对列表元素进行排序，可以对key进行设置(lambda表达式)自定义排序 |



#### 字典

字典是一种可变容器模型，且可存储任意类型对象。字典的每个键值 key=>value 对用冒号 : 分割，每个对之间用逗号(,)分割，整个字典包括在花括号 {} 中。格式：d = {key1 : value1, key2 : value2, key3 : value3 }

**注意：**

* 字典在数据结构中用于描述哈希表
* dict是 Python 的关键字和内置函数，变量名不建议命名为 dict;
* 键必须是唯一的，但值则不必。值可以取任何数据类型，但键必须是不可变的，如字符串，数字;
* 特殊用法：key in dict 将返回某键是否存在于字典中（高效率），返回True/False

**语法：**

创建：

`dict()`	创建字典（也可以用`d={}`）

属性：

`len()`	计算字典的元素个数

操作：

| 函数                      | 描述                                                      |
| :------------------------ | :-------------------------------------------------------- |
| d.clear()                 | 清空字典                                                  |
| d.copy()                  | 返回字典的浅复制                                          |
| d.get(key,default='None') | 返回指定键的值，如果键不在字典中返回 default 设置的默认值 |
| d.update(dict2)           | 把字典dict2的键/值对更新到dict里                          |
| d.pop(key)                | 删除字典给定键 key 所对应的值，**返回值**为被删除的值。   |

特殊用法：

```python
# 快速创建字典
p = "abc"
p_dict = dict((i,p.count(i)) for i in p)	# {'a': 1, 'b': 1, 'c': 1}
# 直接添加元素
window = {}
for i in p:
    window.get()
```



#### 集合

集合（set）是一个**无序**的**不重复**元素序列。

注意：

* 集合用于表示数据结构中的哈希集
* 创建一个空集合必须用 set() 而不是 {}，{} 是用于一个空字典；
* 特殊用法：key in set 将返回某元素是否存在于集合中（高效率），返回True/False

**语法：**

创建：

`set()`	创建集合，唯一方法

属性：

`len()`	计算集合的元素个数

操作：

| 方法              | 描述                                  |
| :---------------- | :------------------------------------ |
| s.add(element)    | 为集合添加元素element                 |
| s.clear()         | 移除集合中的所有元素                  |
| s.copy()          | 拷贝一个集合，返回集合的浅拷贝        |
| s.remove(element) | 从集合中删除元素element，不存在则报错 |



## 方法

#### 滑动窗口

滑动窗口是一种基于**双指针**的一种思想，两个指针指向的元素之间形成一个窗口。

**应用**：

* 一般给出的数据结构是数组或者字符串

* 求取某个子串或者子序列最长最短等最值问题或者求某个目标值时

* 该问题本身可以通过暴力求解

**步骤：**

1. 初始时，左右指针left,right都指向第0个元素，窗口为[left,right)，注意这里是左闭右开，因此初始窗口[0,0)区间没有元素，符合我们的初始定义
2. 开始循环遍历整个数组元素，判断当前right指针是否超过整个数组的长度，是退出循环，否则执行第3步
3. 然后right指针开始向右移动一个长度，并更新窗口内的区间数据
4. 当窗口区间的数据满足我们的要求时，右指针right就保持不变，左指针left开始移动，直到移动到一个不再满足要求的区间时，left不再移动位置
5. 执行第2步

**code:**

```text
int left = 0, right = 0;

while (right < s.size()) {
    // 增大窗口
    window.add(s[right]);
    right++;
    
    while (window needs shrink) {
        // 缩小窗口
        window.remove(s[left]);
        left++;
    }
}
```

**例题：**

[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters)

[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring)

[438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string)



#### 动态规划

动态规划，英文：Dynamic Programming，简称DP，如果某一问题有很多重叠子问题，使用动态规划是最有效的。

所以动态规划中每一个状态一定是由上一个状态推导出来的，**这一点就区分于贪心**，贪心没有状态推导，而是从局部直接选最优的.

**Debug**

* 写代码之前一定要把状态转移在dp数组的上具体情况模拟一遍，心中有数，确定最后推出的是想要的结果；
* 再写代码，如果代码没通过就打印dp数组，看看是不是和自己预先推导的哪里不一样

**步骤**

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组



**例题**

[509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

