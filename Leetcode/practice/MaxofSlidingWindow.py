'''
@File    :   MaxofSlidingWindow.py
@Time    :   2021/07/22 16:03:44
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

给定数组nums，大小为k的滑动窗口从数组最左端移动到最右端，找出所有滑动窗口的最大值。

解法：使用单调队列[双向队列]（在队尾添加元素，在队首删除元素，始终维护队列的最大值）
'''

def MaxofSlidingWindow(nums,k):
    if not nums:
        return nums
    res = list() # 存储每个滑动窗口的最大值
    q = list() # q存储索引值
    for i in range(len(nums)):
        while not q and nums[q[-1]]<nums[i]:
            q.pop(-1)
        q.append(i)
        if q[0]==i-k:
            q.pop(0)
        if i >= k-1:    # 表示索引值到达窗口长度，可以开始记录窗口的最大值
            res.append(nums[q[0]])
    return res


if __name__ == "__main__":
    a = [1,3,-1,-3,5,3,6,7]
    res = MaxofSlidingWindow(a,3)
    print(res)