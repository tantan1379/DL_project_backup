package database.leetcode_3;

import java.util.HashSet;
import java.util.Set;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_3
 * @Date: 2022/1/11 22:50
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
 */
class Solution {
    public int lengthOfLongestSubstring(String s) {//using sliding window
        int rp = -1;
        Set<Character> hashset = new HashSet<>();
        int res = 0;
        int n = s.length();
        for (int i = 0; i < n; i++) {
            if (i != 0) {
                hashset.remove(s.charAt(i - 1));
            }
            while ((rp + 1 < n) && !hashset.contains(s.charAt(rp + 1))) {
                hashset.add(s.charAt(rp + 1));
                rp++;
            }
            res = Math.max(res, rp - i + 1);
        }
        return res;
    }

    public static void main(String[] args) {
        Solution solution_3 = new Solution();
        int res = solution_3.lengthOfLongestSubstring("qqwetryq");
        System.out.println(res);
    }
}