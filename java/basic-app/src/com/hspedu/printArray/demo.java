package com.hspedu.printArray;

public class demo {
    public static void main(String[] args) {
        int[] array = {1,2,3,4,5};
        printArray(array);
    }

    public static void printArray(int[] array) {
        if (array != null && array.length != 0) {
            System.out.print("[");
            for (int i = 0; i < array.length; i++) {
                System.out.print(i != array.length - 1 ? array[i] + "," : array[i]);
            }
            System.out.println("]");
        }
    }

}
