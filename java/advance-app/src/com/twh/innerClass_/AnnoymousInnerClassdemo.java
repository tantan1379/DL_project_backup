package com.twh.innerClass_;

public class AnnoymousInnerClassdemo {
    public static void main(String[] args) {
        test(new Test(){
            @Override
            public void show() {
                System.out.println("显示内容");
            }
        });
    }

    public static void test(Test t){
        t.show();
    }
}

interface Test{
    void show();
}