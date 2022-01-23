package com.twh.exception_;

import java.util.Scanner;

/**
 * @Project: advance-app
 * @Package: com.twh.exception_
 * @Date: 2022/1/13 16:35
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class repeat_input {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String inputStr = "";
        int num = 0;
        while(true){
            inputStr = scanner.next();
            try{
                num = Integer.parseInt(inputStr);
                break;
            }catch(NumberFormatException e){
                System.out.print("你输入的不是一个整数：");
            }
        }
        System.out.println("输入的值为="+num);
    }
}
