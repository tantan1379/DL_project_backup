package com.hspedu.Recursion;

public class Peach {
    public static void main(String[] args) {
        int a = 1;
        B b = new B();
        System.out.println(b.get_peach(a));
    }
}

class B{
    public int get_peach(int day){
        if(day==10){
            return 1;
        }
        return 2*get_peach(day+1)+1;
    }
}
