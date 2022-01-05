package com.hspedu.polymorphic.basic;

public class Cat extends Animal {
    public Cat() {
    }

    public Cat(String name) {
        super(name);
    }

    public void cry() {
        System.out.println("Cat cry()猫在哭。。");
    }
}
