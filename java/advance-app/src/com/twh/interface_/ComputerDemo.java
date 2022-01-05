package com.twh.interface_;

public class Computer {
    public void work(Usb usb){
        usb.start();
        usb.stop();
    }

    public static void main(String[] args) {
        Computer c = new Computer();
        c.work(new Camera());//这里发生了接口的动态绑定：形参是接口，实参是实现该接口的对象
        c.work(new Phone());
    }
}

interface Usb {
    int a =10;
    int b = 10;

    void start();
    void stop();
    default void test(){

    }
}


class Camera implements Usb{
    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }
}

class Phone implements Usb{
    @Override
    public void start() {
        System.out.println("手机工作");
    }

    @Override
    public void stop() {
        System.out.println("手机停止");
    }

}
