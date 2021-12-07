package com.hspedu.method;

public class varParameter {
    public static void main(String[] args) {
        Tools t = new Tools();
        int mySum = t.getSum(10,12,1);
        System.out.println(mySum);
    }
}

class Tools{
    public int getSum(int... nums){
        int sum = 0;
        for(int i=0;i<nums.length;i++){
            sum += nums[i];
        }
        return sum;
    }
}
