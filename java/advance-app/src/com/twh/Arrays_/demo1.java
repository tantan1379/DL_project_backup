package com.twh.Arrays_;

import java.util.Arrays;
import java.util.Comparator;

/**
 * @Project: advance-app
 * @Package: com.twh.Arrays_
 * @Date: 2022/1/17 23:04
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo1{
    public static void main(String[] args) {
        Person[] personArray = new Person[5];
        personArray[0] = new Person("a",18);
        personArray[1] = new Person("b",17);
        personArray[2] = new Person("c",18);

//        Arrays.sort(personArray, new Comparator<>() {
//            @Override
//            public int compare(Person o1, Person o2) {
//                return o1.getAge()-o2.getAge();
//            }
//        });

        System.out.println(Arrays.toString(personArray));
    }
}



class Person{
    private String name;
    private int age;

    public Person(String name, int age) {
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
        return "Person{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
