package com.twh.string_;

/**
 * @Project: advance-app
 * @Package: com.twh.string_
 * @Date: 2022/1/17 21:37
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class exercise1 {
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder("123231243124142214.123123");
        int pointIndex = sb.lastIndexOf(".");
        int i = pointIndex;
        i-=3;
        while(i>0){
            sb.insert(i,",");
            i-=3;
        }
        System.out.println(sb);

    }
}
