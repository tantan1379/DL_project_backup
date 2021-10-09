'''
@File    :   9. Palindrome Number.py
@Time    :   2021/10/07 16:20:24
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
    
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。

示例 1：
输入：x = 121
输出：true

示例 2：
输入：x = -121
输出：false
解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。

示例 3：
输入：x = 10
输出：false
解释：从右向左读, 为 01 。因此它不是一个回文数。

示例 4：
输入：x = -101
输出：false
 
提示：
-2**31 <= x <= 2**31 - 1
''' 

INT_MAX = 2**31-1
INT_MIN = -2**31


def isPalindrome(x):
    """
    :type x: int
    :rtype: bool
    """
    num = x
    reverse_num = 0
    if x<0 or (x>0 and x%10==0):
        return False
    while(x):
        reverse_num = reverse_num*10+x%10
        x = (x-x%10)/10
        if reverse_num>INT_MAX:
            return False
    return reverse_num==num