package com.twh.generic_;

/**
 * @Project: advance-app
 * @Package: com.twh.generic_
 * @Date: 2022/1/22 16:21
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class detail2 {
    public static void main(String[] args) {

    }
}


interface IUsb<T,R>{
    void test(T t,R r);
}

interface IA extends IUsb<String,Double>{

}

class impleIA implements IA{
    @Override
    public void test(String s, Double aDouble) {

    }
}