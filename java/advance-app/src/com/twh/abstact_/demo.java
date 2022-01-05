package com.twh.abstact_;

public class demo {
    public static void main(String[] args) {
        Animal cat = new Cat("mimi","male",10);
        cat.cry();
    }
}

abstract class Animal{
    private String name;
    private String sex;
    private int age;

    public Animal(String name, String sex, int age) {
        this.name = name;
        this.sex = sex;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public abstract void cry();
}

class Cat extends Animal{

    public Cat(String name, String sex, int age) {
        super(name, sex, age);
    }


    @Override
    public void cry() {
        System.out.println(getName()+" is crying...");
    }
}

