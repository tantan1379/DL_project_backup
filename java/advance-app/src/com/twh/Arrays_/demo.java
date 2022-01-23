package com.twh.Arrays_;

import java.util.Arrays;
import java.util.Comparator;

/**
 * @Project: advance-app
 * @Package: com.twh.Arrays_
 * @Date: 2022/1/17 22:46
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo {
    public static void main(String[] args) {
        Integer[] array = {1,3,4,5};
        Arrays.sort(array,new Comparator(){
            @Override
            public int compare(Object o1, Object o2) {
                return 0;
            }
        });
    }
}
