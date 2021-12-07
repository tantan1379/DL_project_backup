package com.hspedu.Recursion;

public class Fibonacci  {
    public static void main(String[] args) {
        A a = new A();
        System.out.println(a.F(7));
    }
}

class A{
    public int F(int n){
        if(n>=1) {
            if (n == 1 || n == 2) {
                return 1;
            }
            else{
                return F(n-1)+F(n-2);
            }
        }
        else{
            System.out.println("Invalid input");
            return -1;
        }
    }
}
