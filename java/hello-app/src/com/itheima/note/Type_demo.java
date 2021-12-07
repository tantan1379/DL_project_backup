package com.itheima.note;
import java.util.Scanner;

public class Type_demo {
    public static void main(String[] args) {
//        Scanner scan = new Scanner(System.in);
//
//        System.out.println("next:");
//        if(scan.hasNext()){
//            String str1 = scan.next();
//            System.out.println(str1);
//        }
//        scan.close();
        char grade = 'B';

        switch (grade) {
            case 'A' -> System.out.println("优秀");
            case 'B', 'C' -> System.out.println("良好");
            case 'D' -> System.out.println("及格");
            case 'F' -> System.out.println("你需要再努力努力");
            default -> System.out.println("未知等级");
        }
        System.out.println("你的等级是 " + grade);
    }
}
