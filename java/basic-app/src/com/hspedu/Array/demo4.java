package com.hspedu.Array;

import java.util.Scanner;

public class demo4 {
    public static void main(String[] args) {
        int[] originArray = {10, 12, 45, 90};
        int[] addArray = new int[originArray.length + 1];
        int index = originArray.length;

        Scanner scan = new Scanner(System.in);
        System.out.print("请输入需要插入的数字：");
        int inputNum = scan.nextInt();

        for (int i = 0; i < originArray.length; i++) {
            if (originArray[i] >= inputNum) {
                index = i;
                break;
            }
        }

//        //自创(复杂版)
//        for (int i = 0; i < addArray.length; i++) {
//            if (i < index) {
//                addArray[i] = originArray[i];
//            } else if(i==index){
//                addArray[i] = inputNum;
//            }else{
//                addArray[i] = originArray[i-1];
//            }
//        }
        //老程序员简洁版
        for (int i = 0, j = 0; i < addArray.length; i++) {
            if(i!=index){
                addArray[i] = originArray[j];
                j++;
            }
            else addArray[i] = inputNum;
        }

        System.out.print("添加元素后的数组为：");
        for (int i = 0; i < addArray.length; i++) {
            System.out.print(addArray[i] + " ");
        }
    }
}
