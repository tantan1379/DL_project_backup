package com.twh.homework;

import java.sql.PreparedStatement;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/18 16:30
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo10 {
    public static void main(String[] args) {
        String name = "Han shun Ping";
        System.out.println(getFormatName(name));
    }

    public static String getFormatName(String name){
        String[] stringArray = name.split(" ");
        return String.format("%s,%s .%c",stringArray[2],stringArray[0],stringArray[1].toUpperCase().charAt(0));
    }
}
