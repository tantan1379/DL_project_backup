package com.twhupup.demo2;

/**
 * @Project: Thread
 * @Package: demo
 * @Date: 2022/3/31 16:23
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestDaemon {
    public static void main(String[] args) {
        God god = new God();
        You you = new You();

        Thread thread = new Thread(god);
        thread.setDaemon(true);

        thread.start();
        new Thread(you).start();

    }
}

class You implements Runnable{
    @Override
    public void run() {
        for (int i = 0; i < 36500; i++) {
            System.out.println("活着很开心");
        }
        System.out.println("goodbye=========");
    }
}

class God implements Runnable{
    @Override
    public void run() {
        while(true){
            System.out.println("上帝保佑你");
        }
    }
}