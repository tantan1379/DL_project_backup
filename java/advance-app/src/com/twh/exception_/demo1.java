package com.twh.exception_;

import java.nio.file.FileAlreadyExistsException;

/**
 * @Project: advance-app
 * @Package: com.twh.exception_
 * @Date: 2022/1/13 21:43
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo1 {
    public static void main(String[] args) {


    }

    public static void f1(){
        try {
            f2();
        } catch (FileAlreadyExistsException e) {
            e.printStackTrace();
        }
    }

    public static void f2() throws FileAlreadyExistsException {

    }

    public static void f3() {
        f4();
    }

    public static void f4() throws RuntimeException {

    }


}
