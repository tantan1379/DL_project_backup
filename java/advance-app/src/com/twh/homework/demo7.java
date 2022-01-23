package com.twh.homework;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/12 22:37
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo7 {
    public static void main(String[] args) {
        Car car = new Car(45);
        Car car1 = new Car(20);
        Car car2 = new Car(-1);
        car.getAir().flow();
        car1.getAir().flow();
        car2.getAir().flow();
//        car.new Air().flow();
//        car1.new Air().flow();
//        car2.new Air().flow();
    }
}

class Car{
    private final int temp;

    public Car(int temp) {
        this.temp = temp;
    }

    class Air{
        public void flow(){
            if(temp>40){
                System.out.println("吹冷气");
            }else if(temp<0){
                System.out.println("吹热气");
            }else{
                System.out.println("关闭空调");
            }
        }
    }

    public Air getAir(){
        return new Air();
    }
}