package com.company;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/6 16:15
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Dep {
    private String name;

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "Dep{" +
                "name='" + name + '\'' +
                '}';
    }
}
