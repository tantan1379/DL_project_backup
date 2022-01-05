package com.twh.block_;

public class demo {
    public static void main(String[] args) {
//        A a = new A(11);
//        A a1 = new A(20,23,40);
//        A.test();
//        new A();
        B b = new B();
    }
}

class A{
    static int n1 = getN1();
    int n2 = getN2();
    int n3 = 30;

    static {
        System.out.println("A的静态代码块");
    }

    {
        System.out.println("A的普通代码块");
    }

    private static int getN1(){
        System.out.println("A静态属性初始化被调用");
        return 40;
    }

    private int getN2(){
        System.out.println("A普通属性初始化被调用");
        return 10;
    }

    public A() {
        //实际隐藏super()【Object】
        //实际隐藏普通代码块和普通属性初始化
        System.out.println("A的构造器");
    }

    public A(int n1) {
        System.out.println("n1");
        A.n1 = n1;
    }

    public A(int n1, int n2, int n3) {
        System.out.println("n1,n2,n3");

        A.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
    }

    public static void test(){
        System.out.println("test");
    }
}

class B extends A{
    static int n5 = getN5();

    private static int getN5(){
        System.out.println("B静态属性初始化被调用");
        return 30;
    }

    private static int getN6(){
        System.out.println("B普通属性初始化被调用");
        return 60;
    }

    static {
        System.out.println("B的静态代码块");
    }

    {
        System.out.println("B的普通代码块");
    }

    public B(){
        //实际隐藏super()
        //实际隐藏{}和普通属性初始化
        System.out.println("B的构造器");
    }

    int n6 = getN6();

}