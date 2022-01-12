package com.twh.enum_;

/**
 * @Project: advance-app
 * @Package: com.twh.enum_
 * @Date: 2022/1/10 13:58
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
@SuppressWarnings({"all"})
public class demo_enum {
    public static void main(String[] args) {
//        System.out.println(Season1.SPRING);
        Season1 s = Season1.SPRING;
//        System.out.println(Season1.values()[1]);
        Season1[] values = Season1.values();
        for (Season1 value : values) {
            System.out.println(value);
        }

        Season1 s1 = Season1.SUMMER;
        System.out.println(s1.compareTo(s));
        System.out.println(Season1.SPRING);
        Season1.SPRING.printInfo();
    }
}

interface Info{
    int a = 0;
    void printInfo();
}


enum Season1 implements Info{
    SPRING("春天"),SUMMER("夏天");

    private final String pname;

    private Season1(String name){
        this.pname = name;
    }

    public String getName() {
        return pname;
    }

    @Override
    public void printInfo() {
        System.out.println("出去玩");
    }

//    @Override
//    public String toString() {
//        return "Season{" +
//                "name='" + getName() + '\'' +
//                '}';
//    }
}