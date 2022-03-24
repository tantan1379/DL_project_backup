package database.leetcode_28;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_28
 * @Date: 2022/2/25 16:38
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
//朴素算法
public class Solution2 {
    public int strStr(String haystack, String needle) {
        char[] ss = haystack.toCharArray();
        char[] pp = needle.toCharArray();
        boolean flag = true;

        for(int i=0;i<ss.length-pp.length+1;i++){
            for(int j=0;j<pp.length;j++){
                if(ss[i+j]!=pp[j]){
                    flag = false;
                    break;
                }
            }
            if(flag){
                return i;
            }
        }
        return -1;
    }
}
