package com.hspedu.Constructor;

public class demo {
    public static void main(String[] args) {
        Person p1 = new Person();
        Person p2 = new Person("tan",12);
        System.out.println(p1.name+" "+p1.age);
        System.out.println(p2.name+" "+p2.age);
//        p1.printName();
//        p1.printAge();
//        p2.printName();
//        p2.printAge();
    }
}

class Person{
    String name;
    int age;

    Person(){
        age = 10;
    }
    Person(String pName,int pAge){
        name = pName;
        age = pAge;
    }

    public void printName(){
        System.out.println("名字是："+this.name);
    }
    public void printAge(){
        System.out.println("年龄是："+this.age);
    }
}