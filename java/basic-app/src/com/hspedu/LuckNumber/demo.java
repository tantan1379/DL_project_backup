package com.hspedu.LuckNumber;

import java.util.Random;
import java.util.Scanner;

public class demo {
    public static void main(String[] args) {
        int[] luckNumbers = createLuckNumber();
        int[] inputNumber = userInputNumber();
        System.out.println("The luckNumber is:");
        for (int num : luckNumbers) {
            System.out.print(num+" ");
        }
        System.out.println("\n");
        System.out.println("The Number you bought is:");
        for (int num : inputNumber) {
            System.out.print(num+" ");
        }
        System.out.println("\n");
        judgeWin(luckNumbers,inputNumber);
    }

    public static int[] createLuckNumber() {
        int[] numbers = new int[7];
        Random rand = new Random();

        for (int i = 0; i < numbers.length - 1; i++) {
            while (true) {
                int data = rand.nextInt(33) + 1;
                boolean flag = true;
                //遍历已经添加的元素，如果有重复则返回；没有重复则加入
                for (int j = 0; j < i; j++) {
                    if (numbers[j] == data) {
                        flag = false;
                        break;
                    }
                }
                //如果该数组未出现过，则添加到数组中
                if (flag) {
                    numbers[i] = data;
                    break;
                }
            }
        }
        numbers[numbers.length - 1] = rand.nextInt(16) + 1;
        return numbers;
    }

    public static int[] userInputNumber() {
        int[] numbers = new int[7];
        Scanner scan = new Scanner(System.in);
        for (int i = 0; i < numbers.length-1; i++) {
            System.out.print("请您输入第" + (i + 1) + "个红球号码（1-33），要求不重复:");
            int data = scan.nextInt();
            numbers[i] = data;
        }
        System.out.print("请您输入蓝球号码(1-16)：");
        numbers[6]= scan.nextInt();
        return numbers;
    }

    public static void judgeWin(int[] luckNumbers, int[] inputNumbers) {
        int count = 0;
        for (int luckNumber : luckNumbers) {
            for (int inputNumber : inputNumbers) {
                if (luckNumber == inputNumber) {
                    count++;
                    break;
                }
            }
        }
        System.out.println("您命中了"+count+"个球");
    }
}


