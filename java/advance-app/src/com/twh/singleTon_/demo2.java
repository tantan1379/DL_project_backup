package com.twh.singleTon_;

public class demo2 {
    public static void main(String[] args) {
        System.out.println(Cat.n1);
        Cat instance = Cat.getInstance();
        Cat instance2 = Cat.getInstance();

    }
}


class Cat{
    private String name;
    public static int n1 = 999;
    private static Cat cat;

    private Cat(String name){
        System.out.println("构造器被调用");
        this.name = name;
    }

    public static Cat getInstance(){
        if(cat==null){
            cat = new Cat("miaomiao");
        }
        return cat;
    }
}