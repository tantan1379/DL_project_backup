package com.twh.exception_;

import java.util.Scanner;

/**
 * @Project: advance-app
 * @Package: com.twh.exception_
 * @Date: 2022/1/14 9:55
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo3 {
    public static void main(String[] args) {
        try {
//            if(args.length!=2){
//                throw new ArrayIndexOutOfBoundsException("参数个数不符");
//            }
            int n1 = Integer.parseInt(args[0]);
            int n2 = Integer.parseInt(args[1]);
            double res = cal(n1,n2);
            System.out.println(res);

        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println(e.getMessage());
        } catch (NumberFormatException e) {
            System.out.println("参数格式错误");
        }catch (ArithmeticException e){
            System.out.println("出现除0错误");
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }

    public static double cal(double n1,int n2){
        return n1/n2;
    }
}


