package com.twh.Collection_.List_;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @Project: advance-app
 * @Package: com.twh.Collection_.ArrayList_
 * @Date: 2022/1/19 21:41
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class demo1 {
    public static void main(String[] args) {
        List<Integer> arrayList = new ArrayList<>();
        arrayList.add(1);
        arrayList.add(3);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
        arrayList.add(4);
//        Iterator<Integer> iterator = arrayList.iterator();
//        while (iterator.hasNext()) {
//            Integer next = iterator.next();
//            System.out.println(next);
//        }
        System.out.println(arrayList);

//        System.out.println(arrayList);


        Iterator iter = arrayList.listIterator();
        Object o = iter.next();
//        iter.remove();


//        while (iter.hasNext()) {
//            Object next = iter.next();
//            System.out.print(next+" ");
//        }
    }

}
