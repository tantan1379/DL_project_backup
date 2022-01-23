package com.twh.generic_;

/**
 * @Project: advance-app
 * @Package: com.twh.generic_
 * @Date: 2022/1/22 13:12
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class Custom {
    public static void main(String[] args) {
        Tiger<String, Integer, Double> t = new Tiger<>("tigerking");
        t.setM(100.0);
        System.out.println(t.m.getClass());

        Tiger<String, Integer, Double> t1 = new Tiger<>("qwe");
        t1.setM(200.0);
        System.out.println(t.m.getClass());
    }
}

class Tiger<T,R,M>{
    String name;
    R r;
    T t;
    M m;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public R getR() {
        return r;
    }

    public void setR(R r) {
        this.r = r;
    }

    public T getT() {
        return t;
    }

    public void setT(T t) {
        this.t = t;
    }

    public M getM() {
        return m;
    }

    public void setM(M m) {
        this.m = m;
    }

    public Tiger(String name) {
        this.name = name;
    }
}
