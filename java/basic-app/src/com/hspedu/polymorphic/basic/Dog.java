package com.hspedu.polymorphic;

public class Dog extends Animal{
    public Dog() {
    }

    public Dog(String name) {
        super(name);
    }

    public void cry() {
        System.out.println("Dog cry() 小狗在哭。。");
    }
}
