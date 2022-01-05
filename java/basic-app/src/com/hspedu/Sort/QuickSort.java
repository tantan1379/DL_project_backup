package com.hspedu.Sort;

public class QuickSort {
    public static void main(String[] args) {
        double[] array = {11.1,13,8.2,4,9.5};
        System.out.println("Before sorting:");
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]+" ");
        }
        System.out.println();
        System.out.println("After sorting:");
        Sort.quickSort(array,0,array.length-1);
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i]+" ");
        }
    }
}

class Sort{
    public static void quickSort(double[] array,int L,int R){
        int left = L;
        int right = R;
        double pivot = array[left];
        if(L>=R){
            return;
        }

        while(left<right){
            while(left<right && array[right]>=pivot){
                right--;
            }
            array[left] = array[right];
            while(left<right && array[left]<=pivot){
                left++;
            }
            array[right] = array[left];
        }
        array[left] = pivot;
        quickSort(array,left+1,R);
        quickSort(array,L,right-1);
    }
}
