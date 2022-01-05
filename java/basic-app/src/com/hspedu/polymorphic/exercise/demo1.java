package com.hspedu.polymorphic.exercise;

public class demo1 {
    public static void main(String[] args) {
        double d = 13.4;
        long l = (long)d;//double比long的范围更大，大范围的数据可以转换为小范围的数据
        int in = 5;
        //boolean b = (boolean)in;//报错，int无法转为boolean，如需要转换使用三目运算符
        Object obj = "hello";//向上转型：将父类引用(obj)指向子类对象（String的"hello")
        String objStr = (String)obj;//向下转型：将父类引用强制转换为子类引用，父类引用指向的String是当前目标类型

        Object objPri = new Integer(5);//向上转型：将父类引用(Object的obj)指向子类对象（Integer的5)
        //String str = (String)objPri;//报错，因为强转过程中父类引用未指向原有的子类对象(Integer)
        Integer str1 = (Integer)objPri;//向下转型：将父类引用(objPri)强制转换为子类引用(str1),父类引用指向的Integer是原有的目标对象
    }
}
