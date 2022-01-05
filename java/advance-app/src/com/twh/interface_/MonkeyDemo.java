package com.twh.interface_;

public class MonkeyDemo {
    public static void main(String[] args) {
        littleMonkey wukong = new littleMonkey("wukong");
        wukong.climbing();
        wukong.fly();
        wukong.swim();
    }
}

class Monkey{
    String name;

    public Monkey(String name) {
        this.name = name;
    }

    public void climbing(){
        System.out.println(name+"猴子会爬树");
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}

class littleMonkey extends Monkey implements Birdable,Fishable{

    public littleMonkey(String name) {
        super(name);
    }

    @Override
    public void fly() {
        System.out.println(name+"通过学习，可以像鸟儿一样飞行");
    }

    @Override
    public void swim() {
        System.out.println(name+"通过学习，可以像鱼儿一样有用");
    }
}

interface Birdable{
    void fly();
}

interface Fishable{
    void swim();
}