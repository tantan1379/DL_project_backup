package com.heima.Array;

public class print2DArray {
    public static void main(String[] args) {
        int[][] array = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                System.out.print(array[i][j]);
                if(j!=array[i].length-1)  System.out.print(",\t");
            }
            System.out.println();
        }
    }
}
