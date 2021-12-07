package com.itheima.random;

import java.util.Random;

public class demo {
    public static void main(String[] args) {
        Random r = new Random();
        for(int i=0;i<20;i++){
            int randomNum = r.nextInt(9);
            System.out.println(randomNum);
        }
    }
}
