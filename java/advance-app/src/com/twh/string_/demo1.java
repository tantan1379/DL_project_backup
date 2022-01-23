package com.twh.string_;


/**
 * @Project: advance-app
 * @Package: com.twh.String
 * @Date: 2022/1/14 14:53
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo1 {
    public static void main(String[] args) {
        int[] aArray = {1,2,3};
        System.out.println(aArray.length);

        String  aString = " ABC   ";
        System.out.println(aString.trim());
        char[] aCharArray = aString.toCharArray();

        for (int i = 0; i < aCharArray.length; i++) {
            System.out.println(aCharArray[i]);
        }
    }
}
