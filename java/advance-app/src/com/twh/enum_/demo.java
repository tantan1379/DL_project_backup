package com.twh.enum_;

/**
 * @File :   train.py
 * @Time :   2021/08/12 10:23:52
 * @Author :   Tan Wenhao
 * @Version :   1.0
 * @Contact :   tanritian1@163.com
 * @License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
 */
public class demo {
    public static void main(String[] args) {
        System.out.println(Season.SPRING.getName());
        System.out.println(Season.SUMMER);
    }
}

class Season{
    private final String name;
    public static final Season SPRING = new Season("春天");
    public static final Season SUMMER = new Season("夏天");

    private Season(String name){
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return "Season{" +
                "name='" + getName() + '\'' +
                '}';
    }
}