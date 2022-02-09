package com.company;

/**
 * @Project: spring_demo
 * @Package: com.company
 * @Date: 2022/2/6 16:14
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Emp {
    private String name;
    private int age;
    private Dep dep;

    public void setName(String name) {
        this.name = name;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public void setDep(Dep dep) {
        this.dep = dep;
    }

    @Override
    public String toString() {
        return "Emp{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", dep=" + dep +
                '}';
    }
}
