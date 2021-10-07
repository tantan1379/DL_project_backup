'''
Given a string s, return the longest palindromic substring in s. 回文字符串
'''

# # ----------------------------------------------------------------
# # Dynamic planning O(n^2)
# def longestPalindrome(s):
#     n = len(s)
#     if n <= 1:
#         return s
#     max_len = 1
#     begin = 0
#     dp = [[False]*n for _ in range(n)]
#     for i in range(n):
#         dp[i][i] = True

#     for L in range(2, n+1):
#         for i in range(n):
#             j = i+L-1
#             if j>=n:
#                 break
#             if s[i]==s[j]:
#                 if j-i<3:
#                     dp[i][j]=True
#                 else:
#                     dp[i][j]=dp[i+1][j-1]
#             if dp[i][j] and j-i+1>max_len:
#                 maxlen = j-i+1
#                 begin = i
#     return s[begin:begin+maxlen]


# # expand from center O(n^2)
def expand_from_center(s,left,right):
    while left>=0 and right<len(s) and s[left]==s[right]:
        left -= 1
        right += 1
    # 有一定技巧，如果left和right不从一个点出发：
    # 倘若一开始left就不等于right，则left和right交换位置，相减为-1，不构成新的子回文串；
    # 倘若left一开始等于right，则最后一次会比应限定的回文子串往外多扩展一位，需要扣除。
    # 如果left和right从头一点出发：
    # 则无论如何，都会多执行一遍扩展，需要扣除。
    return left+1, right-1  

def longestPalindrome(s):
    begin,end = 0,0
    for i in range(len(s)):
        left1,right1 = expand_from_center(s,i,i)
        left2,right2 = expand_from_center(s,i,i+1)
        print(left2,right2)
        if right1-left1>end-begin:
            end = right1
            begin = left1
        if right2-left2>end-begin:
            end = right2
            begin = left2
    
    return s[begin:end+1]

# # violence O(n^2)
# def isPalindrome(sub):
#     left = 0
#     right = len(sub)-1
#     for i in range(len(sub)):
#         if sub[left]!=sub[right]:
#             return False
#         else:
#             left+=1
#             right-=1
#     return True 

# def longestPalindrome(s):
#     """
#     :type s: str
#     :rtype: str
#     """
#     max_len = 1
#     if(len(s)<2):
#         return s
#     for i in range(len(s)):
#         for j in range(i+1,len(s)+1):
#             sub = s[i:j]
#             # print(sub)
#             if isPalindrome(sub) and len(sub)>max_len:
#                 max_len = len(sub)
#                 ans = sub
#     return ans


if __name__ == "__main__":
    print(longestPalindrome("cABBAc"))