package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 20:35
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo2 {
    public static void main(String[] args) {
        Frock frock = new Frock();
        Frock frock2 = new Frock();
        Frock frock3 = new Frock();
        System.out.println(frock.getSerialNumber());
        System.out.println(frock2.getSerialNumber());
        System.out.println(frock3.getSerialNumber());
    }
}

class Frock{
    private static int currentNum=100000;//类对象共享
    private final int serialNumber;//每个对象独有
    private static boolean flag = false;

    public Frock() {
        this.serialNumber = getNextNum();
    }

    public static int getNextNum(){
        if(flag) {
            currentNum += 100;
        }
        flag = true;//跳过第一次增加
        return currentNum;
    }

    public int getSerialNumber() {
        return serialNumber;
    }
}
