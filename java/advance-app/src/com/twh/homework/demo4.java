package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 20:53
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo4 {
    public static void main(String[] args) {
        Cellphone cellphone = new Cellphone();
        ICaculate iadd = new ICaculate() {
            @Override
            public double work(double n1, double n2) {
                return n1+n2;
            }
        };
        ICaculate imul = new ICaculate() {
            @Override
            public double work(double n1, double n2) {
                return n1*n2;
            }
        };

        cellphone.testWork(iadd,10,12);
        cellphone.testWork(imul,10,12);
    }
}

interface ICaculate{
    double work(double n1, double n2);
}

class Cellphone{
    public void testWork(ICaculate i,int n1,int n2){
        double res = i.work(n1,n2);
        System.out.println("结果为："+res);
    }
}