package com.hspedu.polymorphic.details;

public class PolyDetaildemo {
    public static void main(String[] args) {
        Base base = new Son1();
        Son1 son1 = (Son1)base;
        son1.A();
        son1.B();
        son1.D();
    }
}
