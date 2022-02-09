package com.company;

import sun.misc.Contended;

import java.util.concurrent.CountDownLatch;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/6 22:40
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class CacheLinePadding {
    private static final long COUNT = 10_0000_0000L;

    @Contended
    private static class T {
//        private long p1,p2,p3,p4,p5,p6,p7;
        public long x = 0L;
//        private long p9,p10,p11,p12,p13,p14,p15;
    }

    private static final T[] arr = new T[2];

    static {
        arr[0] = new T();
        arr[1] = new T();
    }

    public static void main(String[] args) throws Exception {

        CountDownLatch latch = new CountDownLatch(2);

        Thread t1 = new Thread(() -> {
            for (long i = 0; i < COUNT; i++) {
                arr[0].x = i;
            }
            latch.countDown();
        });
        Thread t2 = new Thread(() -> {
            for (long i = 0; i < COUNT; i++) {
                arr[1].x = i;
            }
            latch.countDown();
        });
        final long start = System.nanoTime();
        t1.start();
        t2.start();
        latch.await();
        System.out.println((System.nanoTime() - start) / 100_0000);
    }
}

