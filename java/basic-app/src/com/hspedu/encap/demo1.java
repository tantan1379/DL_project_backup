package com.hspedu.encap;

public class demo1 {
    public static void main(String[] args) {
        Person p = new Person("tom",40,100000);
        p.setName("jack");
        p.setAge(30);
        p.setSalary(300000);
        System.out.println(p.info());
    }
}

class Person {
    public String name;
    private int age;
    private double salary;

    public Person() {
    }

    public Person(String name, int age, double salary) {
        setName(name);
        setAge(age);
        setSalary(salary);
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public double getSalary() {
        return salary;
    }

    public void setAge(int age) {
        if (age >= 1 && age <= 120) {
            this.age = age;
        } else {
            System.out.println("年龄需要在1-120");
            this.age = 18;
        }
    }

    public void setName(String name) {
        if(name.length()>=2&&name.length()<=6) {
            this.name = name;
        }else{
            System.out.println("名字长度需要在2-6");
            this.name = null;
        }
    }

    public void setSalary(double salary) {
        this.salary = salary;
    }

    public String info() {
        return "name=" + name + " age=" + age + " salary=" + salary;
    }
}
