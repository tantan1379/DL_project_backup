'''
@File    :   14. Longest Common Prefix.py
@Time    :   2021/10/07 16:11:58
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

 
示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
 

提示：

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] 仅由小写英文字母组成

'''

# 横向扫描 O(mn)
def cpf(str1,str2): # 返回两个字符串的公共前缀
    index = 0
    length = min(len(str1),len(str2))
    while index<length and str1[index]==str2[index]:
        index+=1
    return str1[:index]


def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    prefix = strs[0]    # 初始前缀设为第一个字符串
    for i in range(i,len(strs)):
        prefix = cpf(prefix,strs[i])    # 每次利用cpf函数得到新字符串和之前得到的公共前缀的公共前缀
        if not prefix:  # 如果得到的公共前缀为空，直接返回空
            return ""
    return prefix

# # 纵向扫描 O(mn)
def longestCommonPrefix_portrait(strs):
    """
    :type strs: List[str]
    :rtype: str
    """   
    if not strs:
        return ""
    count = len(strs) 
    length = len(strs[0])
    for i in range(length):
        c = strs[0][i]
        for j in range(1,count):
            if len(strs[j])==i or strs[j][i]!=c:
                return strs[0][:i]
    return strs[0]
        

def longestCommonPrefix_python(strs):
    """
    :type strs: List[str]
    :rtype: str
    """   
    res = ""
    if not strs:
        return ""
    for st in zip(*strs):
        if len(set(st))==1:
            res+=st[0]
        else:
            break
    return res


if __name__ == '__main__':
    print(longestCommonPrefix_python(['abc','aqwe','abcd']))