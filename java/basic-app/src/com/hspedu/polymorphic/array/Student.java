package com.hspedu.polymorphic.array;

public class Student extends Person{
    private double score;

    public Student(String name, int age, double score) {
        super(name, age);
        this.score = score;
    }

    @Override
    public String say() {
        return super.say()+"\t"+"score="+score;
    }

    public void study(){
        System.out.println("学生 "+getName()+" 正在上课");
    }
}
