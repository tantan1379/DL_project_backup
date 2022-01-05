package com.hspedu.Sort;
//bubble sort
public class BubbleSort {
    public static void main(String[] args) {
        int[] array = {4,3,1,5,7,2,1,6};
        bubbleSort(array);

        for (int j : array) {
            System.out.print(j + " ");
        }
    }

    private static void bubbleSort(int[] array) {
        for (int i = 0; i < array.length-1; i++) {
            for (int j = 0; j < array.length-i-1; j++) {
                if(array[j]>array[j+1]){
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }

}
