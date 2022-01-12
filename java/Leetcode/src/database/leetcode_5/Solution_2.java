package database.leetcode_5;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_5
 * @Date: 2022/1/12 14:58
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 给你一个字符串 s，找到 s 中最长的回文子串。
 * tips:在获取子串的题目中，大多数情况我们选择定义子串的起始下标和长度，再做截取操作。
 */
//TODO Review
public class Solution_2 {//using dynamic plan
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        boolean[][] dp = new boolean[len][len];//dp各元素默认值为false

        char[] charArray = s.toCharArray();

        for (int i = 0; i < len; i++) {
            dp[i][i] = true;//长度为1的子串一定为回文子串
        }

        int maxlen = 1;
        int begin = 0;//记录最大长度发生的左边界索引

        for (int L = 2; L <= len; L++) {
            for (int i = 0; i < len; i++) {
                int j = i + L - 1;
                if (j >= len) break;//控制循环结束时间

                if (charArray[i] != charArray[j]) {//如果两侧不等
                    dp[i][j] = false;
                } else {//如果两侧相等
                    if (j - i < 3) {//两侧距离为1或2
                        dp[i][j] = true;
                    } else {//两侧距离大于2，则看里面一层是否为回文子串
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                if(dp[i][j]&&j-i+1>maxlen){
                    maxlen = j-i+1;
                    begin = i;
                }
            }
        }
        return s.substring(begin,begin+maxlen);
    }

    public static void main(String[] args) {
        Solution_2 solution_2 = new Solution_2();
        String res = solution_2.longestPalindrome("abccba");
        System.out.println(res);
    }

}
