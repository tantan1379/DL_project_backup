package com.hspedu.object;

public class demo1 {
    public static void main(String[] args) {
//        double[] array = {1.0,2.0,3.0};
//        double[] array = {};
        double[] array = null;
        A01 a = new A01();
        Double res = a.getMax(array);
        if(res!=null){
            System.out.println("array的最大值为"+res);
        }
        else{
            System.out.println("invalid input array");
        }
    }
}

class A01{

    public Double getMax(double[] array){
        if(array!=null && array.length!=0){
            double maxNum = array[0];
            for (int i = 0; i < array.length; i++) {
                if(maxNum<array[i]){
                    maxNum=array[i];
                }
            }
            return maxNum;
        }
        else{
            return null;
        }

    }
}
