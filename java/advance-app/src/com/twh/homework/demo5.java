package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 21:18
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo5 {
    public static void main(String[] args) {
        new A().test();
    }
}

class A{
    private final String name = "xca";
    public void test(){
        class B{
            private final String name = "twh";

            public void show(){
                System.out.println("name = "+A.this.name);
            }
        }
        B b = new B();
        b.show();
    }
}
