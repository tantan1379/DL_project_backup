package com.hspedu.extend;

public class Student {
    public String name;
    public int age;
    private double score;

    public Student() {
        System.out.println("Student无参构造器Student()被调用");
    }

    public Student(String name) {
        System.out.println("Student有参构造器Student(String name)被调用");
    }

    public Student(String name, int age) {
        System.out.println("Student有参构造器Student(String name,int age)被调用");
    }

    public void setScore(double score) {
        this.score = score;
    }

    public double getScore() {
        return score;
    }
}
