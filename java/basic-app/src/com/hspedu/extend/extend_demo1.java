package com.hspedu.extend;

public class extend_demo1 {
    public static void main(String[] args) {
        B b = new B();
    }
}

class A{
    public A(){
        System.out.println("A()被调用");
    }
}

class B extends A{
    public B(){
        this(10);
        System.out.println("B()被调用");
    }
    public B(int a){

        System.out.println("B(int a)被调用");
    }
}