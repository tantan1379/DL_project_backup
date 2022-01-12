package database.leetcode_6;

import java.util.ArrayList;
import java.util.List;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_6
 * @Date: 2022/1/12 16:35
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z字形排列。
 *
 * 比如输入字符串为 "PAYPALISHIRI行数为 3 时，排列如下：
 *
 * P   A   H   N
 * A P L S I I G
 * Y   I   R
 * 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
 *
 * 请你实现这个将字符串进行指定行数变换的函数：
 *
 * string convert(string s, int numRows);
 *
 */
public class Solution {
    public String convert(String s, int numRows) {
        char[] charArray = s.toCharArray();
        if(numRows<2){
            return s;
        }
        List<StringBuilder> list = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            list.add(new StringBuilder());
        }
        int i = 0;
        for(element:s){
            list.get(i).
        }


    }


}
