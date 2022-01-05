package com.twh.static_;

public class demo {
    public static void main(String[] args) {
        staticDemo s1 = new staticDemo();
        staticDemo s2 = new staticDemo();
        staticDemo.addMoney(100);
        staticDemo.addMoney(50);
        staticDemo.showMoney();
        System.out.println("=================");
        System.out.println(staticDemo.money);
        staticDemo staticDemo = new staticDemo();
        staticDemo.test01();
        System.out.println(staticDemo);
    }
}

class staticDemo {
    public static int money = 10;
    public int age = 18;


    public staticDemo() {
    }

    public staticDemo(int age) {
        this.age = age;
    }

    public static void addMoney(int money) {
        staticDemo.money += money;
    }

    public static void showMoney() {
        System.out.println("总金额为：" + staticDemo.money);
    }

    public void test01(){
        class Inner01{
            public void printInfo(){
//                System.out.println(staticDemo.this.age);
                System.out.println(staticDemo.this);
                System.out.println(this);
            }
        }

        Inner01 inner01 = new Inner01();
        inner01.printInfo();
        System.out.println(inner01);
    }

}