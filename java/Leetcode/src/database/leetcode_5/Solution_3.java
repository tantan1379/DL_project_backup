package database.leetcode_5;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_5
 * @Date: 2022/1/12 14:58
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * <p>
 * 给你一个字符串 s，找到 s 中最长的回文子串。
 * tips:在获取子串的题目中，大多数情况我们选择定义子串的起始下标和长度，再做截取操作。
 */

//TODO REVIEW
public class Solution_3 {//using center expanding

    public String longestPalindrome(String s) {
        int len = s.length();
        int maxlen = 0;
        int begin = 0;
        if(len<2){
            return s;
        }
        char[] charArray = s.toCharArray();
        for (int i = 0; i < charArray.length-1; i++) {
            int evenLen = expandfromCenter(charArray,i,i);
            int oddLen = expandfromCenter(charArray,i,i+1);
            int maxLen = Math.max(evenLen,oddLen);
            if(maxLen>maxlen) {
                begin = i-(maxLen-1)/2;
                maxlen = maxLen;
            }
        }

        return s.substring(begin,begin+maxlen);
    }

    public int expandfromCenter(char[] array, int left, int right) {
        while (left >= 0 && right < array.length) {
            if (array[left] == array[right]) {
                left--;
                right++;
            } else break;
        }
        return right-left-1;
    }

    public static void main(String[] args) {
        Solution_3 solution_3 = new Solution_3();
        String res = solution_3.longestPalindrome("abccba");
        System.out.println(res);
    }
}