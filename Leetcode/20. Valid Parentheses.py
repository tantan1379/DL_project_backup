'''
@File    :   20. Valid Parentheses.py
@Time    :   2021/10/07 16:20:46
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
'''

def isValid(s):
        if len(s)%2!=0:
            return False
        mapping = {"}":"{","]":"[",")":"("}
        stack = list()
        for c in s:
            if c in mapping.keys():
                # not stack表示只存在后括号，此时括号无效；stack[-1]!=mapping[c]表示最后一个传入的前括号与当前的后括号不匹配。
                if not stack or stack[-1]!=mapping[c]:
                    return False
                # 括号有效时，就将前半括号pop
                stack.pop()
            else:
                stack.append(c)
        # 循环完所有括号后，如果栈里没有括号，则说明已全部匹配；否则，未匹配完成，括号无效
        return not stack


print(isValid("{{()}[]}"))

