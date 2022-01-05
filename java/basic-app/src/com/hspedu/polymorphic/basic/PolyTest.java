package com.hspedu.polymorphic.basic;

public class PolyTest {
    public static void main(String[] args) {
//        Animal animal = new Dog();//Animal是编译类型 Dog是运行类型
//        animal.cry();//运行时，看animal的运行类型决定该行的执行情况
//        animal = new Cat();
//        animal.cry();
        Master master = new Master("Tom");
        Cat c = new Cat("喵喵");
        Dog d = new Dog("大黄");
        Beef f1 = new Beef("五花肉");
        Pork f2 = new Pork("牛肚");
        master.feed(c,f1);
        master.feed(d,f2);
    }
}
