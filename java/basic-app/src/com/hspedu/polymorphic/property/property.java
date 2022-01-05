package com.hspedu.polymorphic.property;


public class property {
    public static void main(String[] args) {
        Base f = new Sub();
        System.out.println(f.a);
        Sub s = (Sub)f;
        System.out.println(s.a);
        System.out.println(f.test());
        System.out.println(s.test());
    }
}

