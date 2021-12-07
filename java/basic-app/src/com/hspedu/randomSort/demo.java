package com.hspedu.randomSort;

import java.util.Random;
import java.util.Scanner;

public class demo {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        Random r = new Random();
        int[] ids = new int[5];
        for (int i = 0; i < ids.length; i++) {
            System.out.println("请输入员工的工号");
            int id = scan.nextInt();
            ids[i] = id;
        }
        for (int i = 0; i < ids.length; i++) {
            int randomIndex = r.nextInt(ids.length);
            int temp = ids[randomIndex];
            ids[randomIndex] = ids[i];
            ids[i] = temp;
        }

        for (int id : ids) {
            System.out.print(id + "\t");
        }
    }
}
