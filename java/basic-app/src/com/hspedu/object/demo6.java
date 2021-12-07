package com.hspedu.object;

public class demo6 {
    public static void main(String[] args) {

    }
}

class Employee{
    String name;
    int age;
    int salary;

    public Employee(String name){
        this.name = name;
    }
    public Employee(int age){
        this.age = age;
    }
    public Employee(String name, int age, int salary){
        this(name);
        this.salary = salary;
        this.age = age;
    }
}
