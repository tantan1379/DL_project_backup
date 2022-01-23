package com.twh.homework;

import java.util.Scanner;

/**
 * @Project: advance-app
 * @Package: com.twh.homework
 * @Date: 2022/1/18 16:03
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 *
 *
 */
public class demo9 {
    public static void main(String[] args) {

        try {
            userRegister("abc","123512","qweqwe@qwe.c@om");
            System.out.println("注册成功");
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }



    /**
     * 输入用户名、密码、邮箱，如果信息录入正确，则提示注册成功，否则生成异常对象
     * 要求：
     * (1) 用户名长度为2或3或4
     * (2) 密码的长度为6，要求全是数字  isDigital
     * (3) 邮箱中包含@和.   并且@在.的前面
     * <p>
     * 思路分析
     * (1) 先编写方法 userRegister(String name, String pwd, String email) {}
     * (2) 针对 输入的内容进行校核，如果发现有问题，就抛出异常，给出提示
     * (3) 单独的写一个方法，判断 密码是否全部是数字字符 boolean
     */
    public static void userRegister(String name,String pwd,String email){

        if(!(name != null && pwd != null && email != null)) {
            throw  new RuntimeException("参数不能为null");
        }

        if(!(name.length()<6&&name.length()>2)){
            throw new RuntimeException("用户名长度为2或3或4");
        }

        if(!(pwd.length()==6&&isDigital(pwd))){
            throw new RuntimeException("密码的长度为6，要求全是数字");
        }

        int j = email.indexOf('.');
        int i = email.indexOf('@');
        if(!(i>0 && j>i)){
            throw new RuntimeException("邮箱中包含@和.   并且@在.的前面");
        }

    }
    public static boolean isDigital(String str) {
        char[] chars = str.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] < '0' || chars[i] > '9') {
                return false;
            }
        }
        return true;
    }
}
