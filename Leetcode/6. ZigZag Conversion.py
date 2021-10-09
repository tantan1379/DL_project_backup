'''
@File    :   6. ZigZag Conversion.py
@Time    :   2021/10/07 16:20:09
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P   A   H   N
A P L S I I G
Y   I   R

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

请你实现这个将字符串进行指定行数变换的函数：
string convert(string s, int numRows);

'''

def convert(s,numRows):
    """
    :type s: str
    :type numRows: int
    :rtype: str
    """
    rows = [[] for _ in range(numRows)]
    curRow = 0
    converse_flag = False
    if numRows==1:
        return s
    for c in s:
        rows[curRow].append(c)
        if curRow == 0 or curRow == numRows-1:
            converse_flag = not converse_flag
        if converse_flag:
            curRow+=1
        else:
            curRow-=1
    # print(rows)
    res = []
    for one_line in rows:
        res += one_line
    res = "".join(res)
    return res

if __name__ == "__main__":
    print(convert("PAYPALISHIRING",3))