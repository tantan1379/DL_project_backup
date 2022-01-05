package com.twh.abstact_;


public class demo2 {
    public static void main(String[] args) {
        AA aa = new AA();
        BB bb = new BB();
        aa.caculateTime();
        bb.caculateTime();
    }
}

abstract class Template{
    public abstract void job();

    public void caculateTime(){
        long start = System.currentTimeMillis();
        job();
        long end = System.currentTimeMillis();
        System.out.println("执行时间： "+(end-start));
    }
}

class AA extends Template{
    @Override
    public void job() {
        long num = 0;
        for (int i = 0; i < 800000; i++) {
            num+=i;
        }
    }
}

class BB extends Template{
    @Override
    public void job() {
        long num = 1;
        for (int i = 0; i < 800000; i++) {
            num*=i;
        }
    }
}
