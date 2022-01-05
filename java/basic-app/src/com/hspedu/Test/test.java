package com.hspedu.Test;

public class test {
    public static void main(String[] args) {
        A a = new A("xyz",10,100.0);
        System.out.println("hashCode() = "+a.hashCode());
        System.out.println("toString() = "+a.toString());
        System.out.println("a = "+a);

    }
}

class A{
    private String a;
    private int b;
    private double c;

    public A(String a, int b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }

    String test(){
        return getClass().getName();
    }
    String test1(){
        return toString();
    }

    @Override
    public String toString() {
        return "A{" +
                "a='" + a + '\'' +
                ", b=" + b +
                ", c=" + c +
                '}';
    }
}