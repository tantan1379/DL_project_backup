package com.hspedu.object;


public class demo2 {
    public static void main(String[] args) {
        A02 a = new A02();
        String[] array = {"twh","sjy","gs","lm"};
        System.out.println(a.find(array,"twh"));
    }
}

class A02 {

    public int find(String[] array, String arr) {
        for (int i = 0; i < array.length; i++) {
            if (array[i].equals(arr)) {
                return i;
            }
        }
        return -1;
    }
}
