'''
@File    :   1. TwoSum.py
@Time    :   2021/10/07 16:12:21
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
你可以按任意顺序返回答案。

 
示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]
 

提示：
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
只会存在一个有效答案

'''

# 暴力 O(n^2)
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]


# 哈希表 O(n)
def twoSum_hashmap(nums, target):
    mapping = {}
    for i in range(len(nums)):
        diff = target - nums[i]
        if diff in mapping.keys():
            return [mapping[diff],i]
        else:
            mapping[nums[i]]=i
    
# 修改版 （返回多个和为target的元素对）
# 双指针 O(n) 无法返回索引
def twoSum_twoPointer(nums,target):
    res = []
    sorted_nums = sorted(nums)
    lp = 0
    rp = len(nums)-1
    while(rp>lp):
        sum = sorted_nums[lp]+sorted_nums[rp]
        left,right = sorted_nums[lp],sorted_nums[rp]
        if(sum<target):
            lp+=1
        elif(sum>target):
            rp-=1
        else:
            res.append([left,right])
            while lp<rp and sorted_nums[lp]==left:
                lp+=1
            while lp<rp and sorted_nums[rp]==right:
                rp-=1

    return res



if __name__ == "__main__":
    arr = [-1, 3, 4, 5, 2, 9]
    res = twoSum_twoPointer(arr, 8)
    print(res)
