package com.twhupup;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/1 20:34
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Order {
    private String address;
    private String name;

    public void setAddress(String address) {
        this.address = address;
    }

    public void setName(String name) {
        this.name = name;
    }

//    public Order(String address, String name) {
//        this.address = address;
//        this.name = name;
//    }

    @Override
    public String toString() {
        return "Order{" +
                "address='" + address + '\'' +
                ", name='" + name + '\'' +
                '}';
    }
}
