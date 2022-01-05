package com.twh.interface_;

public class InterfacePolyPass {
    public static void main(String[] args) {
        IG ig = new Teacher();//接口的引用可以指向 实现了该接口类的对象实例
        IH ih = new Teacher();//由于Teacher也实现了IH的接口，因此也可以实现向上转型
    }
}

interface IH{
}

interface IG extends IH{//此时实现IG的类也需要实现IH
}

class Teacher implements IG{
}

