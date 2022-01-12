'''
@File    :   demo.py
@Time    :   2021/10/07 11:48:37
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''
# def quick_sort(array, start, end):
#     if start >= end:
#         return
#     mid_data, left, right = array[start], start, end
#     while left < right:
#         while array[right] >= mid_data and left < right:
#             right -= 1
#         array[left] = array[right]
#         while array[left] < mid_data and left < right:
#             left += 1
#         array[right] = array[left]
#     array[left] = mid_data
#     quick_sort(array, start, left-1)
#     quick_sort(array, left+1, end)


# def find(nums):
#     if(len(nums))<3:
#         return None
#     sum_nums = 0
#     for num in nums:
#         sum_nums += num
#     avg = sum_nums/len(nums)
#     distance = []
#     for i in range(len(nums)):
#         distance.append(abs(nums[i]-avg))
#     quick_sort(distance,0,len(distance)-1)
#     print(distance)

def compute(string):
    string+=' '
    res = 0
    nums = []
    all_nums = []
    signs = []
    for s in string:
        if s>='0' and s<='9':
            nums.append(s)
        elif s=='+' or s=='-':
            all_nums.append(nums)
            signs.append(s)
            nums = []
        else:
            all_nums.append(nums)

    for i in range(len(all_nums)):
        
        num=int("".join(all_nums[i]))
        if i==0:
            res+=num
        else:
            if signs[i-1]=='+':
                res += num
            else:
                res -= num
    return res 

        

        
    



if __name__ == '__main__':
    string = "123+321-123"
    print(compute(string))