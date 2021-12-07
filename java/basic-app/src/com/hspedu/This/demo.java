package com.hspedu.This;

public class demo {
    public static void main(String[] args) {
        Person p1 = new Person("t",10);
        Person p2 = new Person("t",10);

        System.out.println(p1.compareTo(p2));
    }
}

class Person{
    String name;
    int age;

    public Person(String name, int age){
        this.name = name;
        this.age = age;
    }

    public boolean compareTo(Person p){
        return p.name.equals(this.name) && p.age == this.age;
    }
}