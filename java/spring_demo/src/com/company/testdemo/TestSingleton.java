package com.company.testdemo;

/**
 * @Project: spring_demo
 * @Package: com.company.testdemo
 * @Date: 2022/2/8 23:32
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestSingleton {
    public static void main(String[] args) {
        for (int i = 0; i < 100; i++) {
            new Thread(() -> {
                System.out.println(Singleton_l.getInstance().hashCode());
            }
            ).start();
        }
    }
}

//class Singleton {
//    public static Singleton singleton = new Singleton();
//
//    private Singleton() {
//        System.out.println("饿汉模式调用构造器");
//    }
//
//    public static Singleton getInstance() {
//        return singleton;
//    }
//}

class Singleton_l {
    public static Singleton_l singleton;

    private Singleton_l() {
        System.out.println("懒汉模式调用构造器");
    }

    public static Singleton_l getInstance() {
        if (singleton == null) {
//            synchronized (Singleton_l.class) {

                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
//            }
//            if (singleton == null) {
                singleton = new Singleton_l();
//            }
        }
        return singleton;
    }
}