package database.leetcode_5;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_5
 * @Date: 2022/1/12 9:48
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 给你一个字符串 s，找到 s 中最长的回文子串。
 * tips:在获取子串的题目中，大多数情况我们选择定义子串的起始下标和长度，再做截取操作。
 */
public class Solution {
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        int maxlen = 1;
        int begin = 0;

        char[] charArray = s.toCharArray();

        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                if (j - i + 1 > maxlen && isPalindrome(charArray, i, j)) {
                    maxlen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxlen);
    }


    public boolean isPalindrome(char[] array, int i, int j) {
        while (i < j) {
            if (array[i] != array[j]) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }


    public static void main(String[] args) {
        Solution s = new Solution();
        String res = s.longestPalindrome("abecceba");
        System.out.println(res);
    }
}
