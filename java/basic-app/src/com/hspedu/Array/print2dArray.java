package com.hspedu.Array;

public class print2dArray {
    public static void main(String[] args) {
        int[][] array = {{1, 2, 3}, {3, 4, 5}};
        MyTools mytool = new MyTools();
        mytool.print2dArray(array);
    }
}

class MyTools {
    public void print2dArray(int[][] array) {
        for (int[] ints : array) {
            for (int j = 0; j < ints.length; j++) {
                System.out.print(ints[j]);
                if (j != ints.length - 1) System.out.print(",\t");
            }
            System.out.println();
        }
    }
}
