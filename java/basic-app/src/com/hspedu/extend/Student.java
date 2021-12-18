package com.hspedu.extend_;

public class Student {
    public String name;
    public int age;
    private double score;

    public Student(){

    }
    public Student(String name){
        System.out.println("Student有参构造器被调用");
    }

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public double getScore(){
        return score;
    }
}
