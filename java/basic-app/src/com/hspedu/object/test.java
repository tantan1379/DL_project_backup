package com.hspedu.object;

public class test {
    public static void main(String[] args) {
        int a =10;
        int b= 20;
        swap(a,b);
        System.out.println(a);
        System.out.println(b);
    }
    static void swap(int a, int b){
        int temp = a;
        a =b;
        b = temp;
    }
}

class AA{


}