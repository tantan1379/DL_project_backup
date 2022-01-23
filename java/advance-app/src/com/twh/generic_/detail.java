package com.twh.generic_;

import java.util.ArrayList;

/**
 * @Project: advance-app
 * @Package: com.twh.generic_
 * @Date: 2022/1/22 15:31
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class detail {
    public static void main(String[] args) {
        C<Object> bc = new C<>(new A());
        ArrayList<A> as = new ArrayList<>();
    }
}

class A{
}

class B{

}

class C<E>{
    E e;
    public C(E e){
        this.e = e;
    }
}

