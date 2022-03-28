package com.twhupup.demo1;

/**
 * @Project: Thread
 * @Package: com.twhupup.demo1
 * @Date: 2022/3/24 21:34
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestThread1 extends Thread{
    @Override
    public void run() {
        //run方法线程体
        for (int i = 0; i < 20; i++) {
            System.out.println("Thread线程"+i);
        }
    }

    public static void main(String[] args) {
        //主线程
        TestThread1 testThread1 = new TestThread1();//创建线程对象
        testThread1.start();//调用start方法开启线程

        for (int i = 0; i < 1000; i++) {
            System.out.println("主线程"+i);
        }

    }
}

