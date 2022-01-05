package com.twh.innerClass_;

public class AnnoymousInnerClass {
    public static void main(String[] args) {
        Outer02 outer02 = new Outer02();
        outer02.method();
    }
}

class Outer02{
    private int n1 = 10;


    public void method(){
        Animal tiger = new Tiger(){
            @Override
            public void cry() {
                System.out.println("老虎叫唤...");
            }
        };
        tiger.cry();
    }
}


interface Animal{
    void cry();
}

class Tiger implements Animal{

    @Override
    public void cry() {

    }
}