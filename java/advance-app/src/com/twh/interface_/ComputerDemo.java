package com.twh.interface_;

public class ComputerDemo {
    public void work(Usb usb){
        usb.start();
        usb.stop();
    }

    public static void main(String[] args) {
        ComputerDemo c = new ComputerDemo();
        c.work(new Camera());//这里发生了多态：形参是接口，实参是实现该接口的对象（可以接受手机的对象也可以接受相机的对象）
        c.work(new Phone());
    }
}

interface Usb {
    void start();
    void stop();
}


class Camera implements Usb{
    @Override
    public void start() {
        System.out.println("相机工作");
    }

    @Override
    public void stop() {
        System.out.println("相机停止");
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
