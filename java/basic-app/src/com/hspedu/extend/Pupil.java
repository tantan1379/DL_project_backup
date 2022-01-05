package com.hspedu.extend;

public class Pupil extends Student{
    public Pupil(){
        super("tom");
    }

    public Pupil(String name) {
        super(name);
    }

    public Pupil(String name, int age, double score){
        //不指定父类的构造器则会自动调用默认构造器super()
        this.name = name;
        this.age = age;
        this.setScore(score);
        System.out.println("Pupil的有参构造器Pupil(String name,int age, double score)被调用");
    }

    public void testing(){
        System.out.println("Pupil "+name+"is testing math.." );
    }

    public void showInfo(){
        System.out.println("Name "+name+" Age "+age+" score "+getScore());
    }
}
