package com.twh.string_;

/**
 * @Project: advance-app
 * @Package: com.twh.string_
 * @Date: 2022/1/17 19:45
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class bufferdemo {
    public static void main(String[] args) {
        StringBuffer stringBuffer = new StringBuffer("abc");
//        stringBuffer.delete(1,2);
        String s = null;

        stringBuffer.replace(1,2,"dqweqwe");
        System.out.println(stringBuffer);

        stringBuffer.append(s);


    }
}
