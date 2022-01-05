package com.twh.singleTon_;

public class demo {
    public static void main(String[] args) {
        SingleTon instance = SingleTon.getInstance();
        System.out.println(SingleTon.n1);
    }
}


class SingleTon{
    public static int n1 = 100;
    private static final SingleTon instance = new SingleTon();

    private SingleTon(){
        System.out.println("构造器被调用");
    }

    public static SingleTon getInstance(){
        return instance;
    }
}