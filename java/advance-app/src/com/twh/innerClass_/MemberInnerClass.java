package com.twh.innerClass_;

public class MemberInnerClass {
    public static void main(String[] args) {
        Outer03 outer03 = new Outer03();
        outer03.test();//外部类访问内部类
        Outer03.Inner inner = outer03.new Inner();//外部其它类访问内部类 方式一
        Outer03.Inner inner1 = outer03.getInstance();//外部其它类访问内部类 方式二
        inner.say();
        inner1.say();
    }
}

class Outer03{
    private int n1 = 10;
    public String name = "tan";
    class Inner{//成员内部类
        public void say(){
            System.out.println("n1 = "+n1+" name = "+name);
        }
    }

    public void test(){
        Inner inner = new Inner();//外部类访问内部类成员
        inner.say();
    }

    public Inner getInstance(){
        return new Inner();
    }
}

