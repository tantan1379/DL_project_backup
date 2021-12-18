package com.hspedu.extend_;

public class Pupil extends Student{

    public Pupil(){
        super("tom");
    }
    public Pupil(String name) {
        super(name);
    }

    public void testing(){
        System.out.println("Pupil "+name+"is testing math.." );
    }

    public void showInfo(){
        System.out.println("Name "+name+" Age "+age+" score "+getScore());
    }
}
