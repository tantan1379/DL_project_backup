package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 21:25
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo6 {
    public static void main(String[] args) {
        Person tang = new Person("唐僧", Factory.getHorse());
        tang.common();
        tang.passRiver();
        tang.common();
        tang.passRiver();
        tang.climb();
    }
}

interface Vehicles{
    void work();
}

@SuppressWarnings({"FieldMayBeFinal"})
class Person{
    private String name;
    private Vehicles vehicles;

    public Person(String name, Vehicles vehicles) {
        this.name = name;
        this.vehicles = vehicles;
    }

    public void passRiver(){
        if(!(vehicles instanceof Boat)) {
            vehicles = Factory.getBoat();//向上转型
        }
        System.out.print(name);
        vehicles.work();//动态绑定（编译类型为Vehicles接口，运行类型为Boat)
    }

    public void common(){
        if(!(vehicles instanceof Horse)) {
            vehicles = Factory.getHorse();//向上转型
        }
        System.out.print(name);
        vehicles.work();
    }

    public void climb(){
        if(!(vehicles instanceof Plane)) {
            vehicles = Factory.getPlane();//向上转型
        }
        System.out.print(name);
        vehicles.work();
    }
}


class Factory{
    public static Horse horse = new Horse();//饿汉式设计模式
    public static Boat boat = new Boat();
    public static Plane plane = new Plane();

    private Factory(){}

    public static Horse getHorse(){
        return horse;
    }
    public static Boat getBoat(){
        return boat;
    }
    public static Plane getPlane(){
        return plane;
    }
}

class Horse implements Vehicles{

    @Override
    public void work() {
        System.out.println("骑马走路");
    }
}

class Boat implements Vehicles{

    @Override
    public void work() {
        System.out.println("坐船过河");
    }
}

class Plane implements Vehicles{

    @Override
    public void work() {
        System.out.println("坐飞机爬山");
    }
}

