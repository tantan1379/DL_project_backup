package com.twh.exception_;

/**
 * @Project: advance-app
 * @Package: com.twh.exception_
 * @Date: 2022/1/13 21:52
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo2 {
    public static void main(String[] args) {
        Person p = new Person();
        try {
            p.setAge(-1);
        } catch (IllegalAgeException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println(p);
    }
}


class Person {
    private int age;


    public void setAge(int age) throws IllegalAgeException {
        if (age < 0) {
            throw new IllegalAgeException("人的年龄不应该为负数");
        }
        this.age = age;
    }

    public String toString() {
        return "age is " + age;
    }
}

class IllegalAgeException extends Exception {
    //默认构造器
    public IllegalAgeException() {
    }
    //带有详细信息的构造器，信息存储在message中
    public IllegalAgeException(String message) {
        super(message);
    }
}