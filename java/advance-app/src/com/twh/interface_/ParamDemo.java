package com.twh.interface_;

public class demo {
    public static void main(String[] args) {
        System.out.println(Ai.a);
    }
}


interface A{
    int a = 10;
}

class Ai implements A{
    public Ai() {
        System.out.println("Ai的构造器被调用");
    }
}