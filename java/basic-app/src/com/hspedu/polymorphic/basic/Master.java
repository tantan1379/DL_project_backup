package com.hspedu.polymorphic;

public class Master {
    private String name;

    public Master(String name) {
        this.name = name;
    }

    public void feed(Animal animal, Food food){
        System.out.println("主人" + this.name+"给"+animal.getName() + "喂" + food.getName());
    }
}
