package com.twhupup;

import java.util.concurrent.CountDownLatch;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/8 13:58
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class ThreadOrder {

    private static int a;
    private static int b;
    private static int x;
    private static int y;

    public static void main(String[] args) throws Exception {
        for (long i = 0; i < Long.MAX_VALUE; i++) {
            a = 0;
            b = 0;
            x = 0;
            y = 0;

            CountDownLatch latch = new CountDownLatch(2);

            Thread thread1 = new Thread(() -> {
                b = 1;
                x = a;
                latch.countDown();
            });

            Thread thread2 = new Thread(() -> {
                a = 1;
                y = b;
                latch.countDown();
            });
            thread1.start();
            thread2.start();
            latch.await();
            if (x == 0 && y == 0) {
                System.out.println("第"+i+"次输出为0");
                break;
            }
        }
    }

}
