package com.hspedu.polymorphic.DynamicBinding;

public class demo {
    public static void main(String[] args) {
        //a 的编译类型 A, 运行类型 B
        A a = new B();//向上转型
        //由于运行类型B中没有sum()，向上寻找后使用父类的sum()；动态绑定机制使得，sum()中的getI()使用运行类型B中的getI()
        System.out.println(a.sum());
        //由于运行类型B中没有sum1()，向上寻找后使用父类的sum1()；属性没有动态绑定，sum()中的i使用就近类型A中的i（在哪用哪）
        System.out.println(a.sum1());
    }
}

class A {//父类
    public int i = 10;
    //动态绑定机制
    public int sum() {//父类sum()
        return getI() + 10;//20 + 10
    }
    public int sum1() {//父类sum1()
        return i + 10;//10 + 10
    }
    public int getI() {//父类getI
        return i;
    }
}

class B extends A {//子类
    public int i = 20;
    public int getI() {//子类getI()
        return i;
    }
}