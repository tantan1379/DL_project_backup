'''
@File    :   16. 3Sum Closest.py
@Time    :   2021/10/13 10:08:06
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

 

示例：

输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
 

提示：

3 <= nums.length <= 10^3
-10^3 <= nums[i] <= 10^3
-10^4 <= target <= 10^4

'''

def threeSumClosest(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    nums = sorted(nums)
    min_distance = 2**31
    for i in range(len(nums)):
        lp = i+1
        rp = len(nums)-1
        while(lp<rp):
            distance = abs(nums[lp]+nums[rp]+nums[i]-target)
            sum = nums[lp]+nums[rp]+nums[i]
            if distance<min_distance:
                min_distance = distance
                res = sum

            if sum<target:
                lp+=1
            elif sum>target:
                rp-=1
            else:
                return target
    return res
            

if __name__ == "__main__":
    print(threeSumClosest([1,1,1,0],-100))