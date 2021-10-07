'''
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

Example 1:

Input: x = 123
Output: 321
Example 2:

Input: x = -123
Output: -321
Example 3:

Input: x = 120
Output: 21
Example 4:

Input: x = 0
Output: 0
'''

def reverse(x):
    """
    :type x: int
    :rtype: int
    """
    res = 0
    while x!=0:
        if res>(2**31)//10 or res<(-2**31-1)//10+1:
            return 0
        digit = x%10
        if x<0 and digit!=0:    # python中对负数的余操作会保持在[0,9]之内
            digit-=10
        res = res*10+digit
        # x = (x-digit)//10       # python中对负数的整除操作会比原结果多进一(-1)，因此先去除余数部分
        x = x//10       # python中对负数的整除操作会比原结果多进一(-1)，因此先去除余数部分
    return res


if __name__ == '__main__':
    print(reverse(-123))