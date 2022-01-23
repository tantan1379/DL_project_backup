package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 22:44
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo8 {
    public static void main(String[] args) {
        Color.RED.show();
        Color.YELLOW.show();
        Color.BLUE.show();
    }
}

enum Color implements IColor{
    RED(255,0,0),BLUE(0,0,255),YELLOW(255,255,0);

    private int redValue;
    private int greenValue;
    private int blueValue;

    private Color(int redValue, int greenValue, int blueValue) {
        this.redValue = redValue;
        this.greenValue = greenValue;
        this.blueValue = blueValue;
    }


    @Override
    public void show() {
        System.out.println("red="+redValue+" green="+greenValue+" blue="+blueValue);
    }
}

interface IColor{
    void show();
}
