package com.twh.innerClass_;

public class demo1 {
    public static void main(String[] args) {
        Outer01 outer01 = new Outer01();
        outer01.m1();
    }
}

class Outer01{
    private int n1 = 100;
    private void n2(){
        System.out.println("outer01 m2()被调用");
    }

    public void m1(){
        final class Inner01{
            public void f1(){
                System.out.println("内部类Inner01的方法f1()被调用");
                System.out.println("显示n1="+n1);
                n2();
            }
        }

        Inner01 inner01 = new Inner01();
        inner01.f1();
    }
}