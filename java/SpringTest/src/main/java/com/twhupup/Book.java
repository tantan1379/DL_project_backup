package com.twhupup;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/1 19:39
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Book {
    private String name;
    private Double price;
    private Float a;

    public void setName(String bName) {
        this.name = bName;
    }

    public void setPrice(Double price) {
        this.price = price;
    }

    public void setA(Float a) {
        this.a = a;
    }

    @Override
    public String toString() {
        return "Book{" +
                "name='" + name + '\'' +
                ", price=" + price +
                ", a=" + a +
                '}';
    }
}
