package com.twh.block_;

public class demo2 {
    public static void main(String[] args) {
        C c = new C();

    }
}

class C{
    public static int n = getN();
    public int m = getM();

    {
        System.out.println("普通代码块被调用");
    }
    static {
        System.out.println("静态代码块被调用");
    }



    public static int getN(){
        System.out.println("getN被调用");
        return 100;
    }

    public int getM(){
        System.out.println("getM被调用");
        return 200;
    }
}