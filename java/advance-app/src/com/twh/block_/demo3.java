package com.twh.block_;

public class demo3 {
    public static void main(String[] args) {

    }
}

class Base{
    protected static void test(){
        System.out.println("base的test被调用");
    }
}

class Sub extends Base{
    protected static void test() {
        System.out.println("sub的test被调用");
    }
}
