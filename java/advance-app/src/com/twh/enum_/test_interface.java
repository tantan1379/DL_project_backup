package com.twh.enum_;

/**
 * @Project: advance-app
 * @Package: com.twh.enum_
 * @Date: 2022/1/10 22:01
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public enum test_interface implements A{
    ;

    @Override
    public void play() {

    }

    @Deprecated
    public String toString(){

        return null;
    }
}


interface A{
    void play();
}