package database.leetcode_1;

import java.util.HashMap;
import java.util.Map;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_1
 * @Date: 2022/1/11 21:51
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 
 * 给定一个整数数组 nums和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那两个整数，并返回它们的数组下标。
 *
 * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
 *
 * 你可以按任意顺序返回答案。
 */
public class Solution {//using hashmap
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> hashmap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target-nums[i];
            if(hashmap.containsKey(diff)){
                return new int[]{i,hashmap.get(diff)};
            } else{
                hashmap.put(nums[i],i);
            }
        }
        return new int[0];
    }


    public static void main(String[] args) {
        Solution solution = new Solution();
        int[] result = solution.twoSum(new int[]{1,2,3},4);
        for (int j : result) {
            System.out.print(j);
            System.out.print(" ");
        }
    }
}

