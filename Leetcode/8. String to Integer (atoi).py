'''
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
 

示例 1：
输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
第 1 步："42"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："42"（读入 "42"）
           ^
解析得到整数 42 。
由于 "42" 在范围 [-2**31, 2**31 - 1] 内，最终结果为 42 。


示例 2：
输入：s = "   -42"
输出：-42
解释：
第 1 步："   -42"（读入前导空格，但忽视掉）
            ^
第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
             ^
第 3 步："   -42"（读入 "42"）
               ^
解析得到整数 -42 。
由于 "-42" 在范围 [-2**31, 2**31 - 1] 内，最终结果为 -42 。


示例 3：
输入：s = "4193 with words"
输出：4193
解释：
第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
             ^
解析得到整数 4193 。
由于 "4193" 在范围 [-2**31, 2**31 - 1] 内，最终结果为 4193 。


示例 4：
输入：s = "words and 987"
输出：0
解释：
第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
         ^
解析得到整数 0 ，因为没有读入任何数字。
由于 0 在范围 [-2**31, 2**31 - 1] 内，最终结果为 0 。


示例 5：
输入：s = "-91283472332"
输出：-2147483648
解释：
第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
          ^
第 3 步："-91283472332"（读入 "91283472332"）
                     ^
解析得到整数 -91283472332 。
由于 -91283472332 小于范围 [-231, 231 - 1] 的下界，最终结果被截断为 -231 = -2147483648 。
'''

INT_MAX = 2**31-1
INT_MIN = -2**31


# 有限状态机（不容易造成代码冗余，但复杂度不变） O(n)
class Automaton:
    def __init__(self):
        self.first_state = 'start'
        self.sign = 1
        self.ans = 0
        self.state_table = {
            "start":["start","signed","in_number","end"],
            "signed":["end","end","in_number","end"],
            "in_number":["end","end","in_number","end"],
            "end":["end","end","end","end"]
        }

    def get_col(self,c):
        if c.isspace():
            return 0
        elif c=="+" or c=="-":
            return 1
        elif c.isdit():
            return 2
        else:
            return 3

    def get(self,c):
        self.state = self.state_table[self.state][self.get_col(c)]
        if self.state=='in_number':
            self.ans = self.ans*10+int(c)
            self.ans = min(self.ans,INT_MAX) if self.sign==1 else min(self.ans,-INT_MIN)
        elif self.state=='sign':
            self.sign = 1 if c=='+' else -1

def myAtoi(s):
    """
    :type s: str
    :rtype: int
    """
    auto = Automaton()
    for c in s:
        auto.get(c)
    return auto.sign * auto.ans


# 直接法 O(n)
def myAtoi_violent(s):
    index = 0
    sign = 1
    res = 0


    # 去除前导空格
    while index<len(s) and s[index]==" ":
        index+=1

    if index==len(s):
        return 0
    # 考虑符号
    if s[index]=="+":
        index+=1
    elif s[index]=="-":
        sign = -1
        index+=1
    
    while index<len(s):
        # 符号后遇到非数字退出循环
        if s[index]<'0' or s[index]>'9':
            break
        res = res*10 + int(s[index])
        index+=1

    res = sign*res
    res = max(min(res,INT_MAX),INT_MIN)
    return res


if __name__ == "__main__":
    print(myAtoi_violent(" "))