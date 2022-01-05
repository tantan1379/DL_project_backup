package com.hspedu.equals;

public class testEqual {
    public static void main(String[] args) {
        Person p = new Person("a", 12);
        Person p1 = new Person("a",12);
        System.out.println(p.equals(p1));
    }
}
