'''
@File    :   quick_sort.py
@Time    :   2021/12/21 10:57:51
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

# L,R 表示对array的L到R进行快排
def quick_sort(array, L, R):
    if L>=R:    # 递归返回条件：如果范围内只有一个元素
        return
    pivot = array[L]    # 设定选定范围最左边的值作为快排的基准值（注意这里的索引并没有具体意义，只看值）
    left, right = L, R  # 定义左右双指针

    while(left<right):  # 判断双指针是否重合
        while(left<right and array[right]>=pivot):  # 判断双指针是否重合，并且右指针的值不小于基准值则右指针向左移动一格 (为什么必须先动右指针：因为记录过左指针的值为pivot因此left的值可以被覆盖)
            right-=1
        array[left] = array[right]                  # 左指针的值大于基准值则将右指针的值赋值为左指针的值（为何不用判断left<right：因为如果left=right做替换相当于不做）

        while(left<right and array[left]<=pivot):   # 判断双指针是否重合，并且左指针的值不大于基准值则左指针向右移动一格
            left+=1
        array[right] = array[left]                  # 左指针的值大于基准值则将右指针的值赋值为左指针的值

    array[left] = pivot             # 将基准值放在双指针重合的位置，这里left和right都一样
    quick_sort(array, right+1, R)   # 对pivot右侧的元素进行快排，这里left和right都一样
    quick_sort(array, L, right-1)   # 对pivot左侧的元素进行快排，这里left和right都一样

 
if __name__ == "__main__":
    array = [1,14,1,11,13,12,15,17,16]
    quick_sort(array,0,len(array)-1)
    print(array)