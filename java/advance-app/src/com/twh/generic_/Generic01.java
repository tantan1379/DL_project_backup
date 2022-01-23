package com.twh.generic_;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;

/**
 * @Project: advance-app
 * @Package: com.twh.generic_
 * @Date: 2022/1/21 10:06
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Generic01 {
    public static void main(String[] args) {
        ArrayList<Dog> arrayList = new ArrayList<>();
        arrayList.add(new Dog("aa",18));
        arrayList.add(new Dog("bb",10));
        arrayList.add(new Dog("cc",12));

        for(Object o:arrayList){
            System.out.print(((Dog)o).getName());
            System.out.print(" ");
            System.out.println(((Dog)o).getAge());
        }

    }

    @Test
    void test(){
        System.out.println("test被调用");
    }
}

class Dog{
    private String name;
    private int age;

    public Dog(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "Dog{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}