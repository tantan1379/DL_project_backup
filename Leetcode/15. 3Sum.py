'''
@File    :   15. 3Sum.py
@Time    :   2021/10/09 15:20:55
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。


示例 1：
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

示例 2：
输入：nums = []
输出：[]

示例 3：
输入：nums = [0]
输出：[]
 

提示：

0 <= nums.length <= 3000
-105 <= nums[i] <= 105
'''
           

def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums = sorted(nums)
    if len(nums)<3 or not nums:
        return []
    ans = []
    for i in range(len(nums)):
        res = []
        if i>0 and nums[i]==nums[i-1]:
            i+=1
            continue
        lp = i+1
        rp = len(nums)-1
        while(lp<rp):
            left = nums[lp]
            right = nums[rp]
            sum = left+right
            if sum<-nums[i]:
                lp+=1
            elif sum>-nums[i]:
                rp-=1
            else:
                res.append([left,right])
                while lp<rp and nums[lp]==left:
                    lp+=1
                while lp<rp and nums[rp]==right:
                    rp-=1
        for twonum in res:
            twonum.append(nums[i])
            ans.append(twonum)
    return ans

        

if __name__ == "__main__":
    print(threeSum([-1,0,1,2,-1,-4]))