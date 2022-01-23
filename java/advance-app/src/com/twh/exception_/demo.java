package com.twh.exception_;

/**
 * @Project: advance-app
 * @Package: com.twh.exception_
 * @Date: 2022/1/13 15:11
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo {
    public static void main(String[] args) {
        A a = new A();
        System.out.println(a.test());
    }
}

class A{
    int b = 10;
    int i = 0;

    int test(){
        try {
            String str = "yes";
            int a = Integer.parseInt(str);
            A object_a = new A();
            object_a = null;
            System.out.println(object_a.b);

        } catch (NullPointerException e) {
            e.printStackTrace();
        } catch (Exception e){
            e.printStackTrace();
        }
        finally {
            System.out.println("finally");
            return i++;
        }
    }
}