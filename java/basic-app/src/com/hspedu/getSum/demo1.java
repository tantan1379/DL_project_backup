package com.hspedu.getSum;

public class demo1 {
    public static void main(String[] args) {
        int n = 10;
        System.out.println(getSum(n));
    }

    public static int getSum(int n){
        int sum = 0;
        while(n>0){
            sum+=n;
            n--;
        }
        return sum;

    }
}
