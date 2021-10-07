'''
给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器。

 

示例 1：



输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
示例 2：

输入：height = [1,1]
输出：1
示例 3：

输入：height = [4,3,2,1,4]
输出：16
示例 4：

输入：height = [1,2,1]
输出：2
 

提示：

n == height.length
2 <= n <= 105
0 <= height[i] <= 104

'''

# violent solution O(n^2)
def maxArea_violent(height):
    """
    :type height: List[int]
    :rtype: int
    """
    volume_arr = []
    for i in range(len(height)):
        for j in range(i+1,len(height)):
            volume = (j-i)*min(height[i],height[j])
            volume_arr.append(volume)

    return max(volume_arr)

# double pointer O(n)
def maxArea_double_pointer(height):
    lp = 0
    rp = len(height)-1
    max_volume = 0
    while (rp-lp):
        volume = (rp-lp)*min(height[lp],height[rp])
        if(height[lp]>=height[rp]):
            rp-=1
        else:
            lp+=1
        max_volume = max(max_volume,volume)

    return max_volume


if __name__ == "__main__":
    print(maxArea_double_pointer([1,8,6,2,5,4,8,3,7]))

