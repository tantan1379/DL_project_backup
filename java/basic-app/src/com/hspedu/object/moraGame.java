package com.hspedu.object;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

public class moraGame {

    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        Map<Integer, String> Num_Punch = new HashMap<>();
        Num_Punch.put(0, "石头");
        Num_Punch.put(1, "剪刀");
        Num_Punch.put(2, "布");
        Map<Integer, String> Num_Win = new HashMap<>();
        Num_Win.put(0, "You win");
        Num_Win.put(1, "Draw");
        Num_Win.put(2, "You lose");
        int i = 0;
        int[][] arr = new int[10][4];
        Person p = new Person();
        while (p.count != 3) {
            if (p.lossCountNum == 2 || p.winCountNum == 2) {
                break;
            }
            System.out.print("你想出(0:石头 1:剪刀 2:布):");
            int inputNum = s.nextInt();
            while (inputNum > 2 || inputNum < 0) {
                System.out.println("输入有误，请重新输入(0:石头 1:剪刀 2:布)：");
                inputNum = s.nextInt();
            }
            p.setTomGuessNum(inputNum);
            int comNum = p.getComGuessNum();
            int isWin = p.Compare();
            arr[i][0] = i + 1;
            arr[i][1] = inputNum;
            arr[i][2] = comNum;
            arr[i][3] = isWin;
            System.out.println("===================================");
            System.out.println("局数\t玩家出拳\t电脑出拳\t输赢情况");
            System.out.println(i + "\t" + Num_Punch.get(inputNum) + "\t\t" + Num_Punch.get(comNum) + "\t\t" + Num_Win.get(isWin));
            System.out.println("===================================");
            System.out.println("\n");
            i++;
        }
        System.out.println("===========结 果 汇 总===========");
        System.out.println("局数\t玩家出拳\t电脑出拳\t输赢情况");
        for (int[] element : arr) {
            if (element[0] == 0) {
                break;
            }
            System.out.println(element[0] + "\t" + Num_Punch.get(element[1]) + "\t\t" + Num_Punch.get(element[2]) + "\t\t" + Num_Win.get(element[3]));
        }
        System.out.println("================================");
        System.out.print("Last Result: ");
        if (p.winCountNum >= 2) {
            System.out.println("You win");
        } else {
            System.out.println("You lose");
        }
    }
}


class Person {
    int comNum;
    int tomNum;
    int winCountNum = 0;
    int lossCountNum = 0;
    int count = 0;

    public int getComGuessNum() {
        Random r = new Random();
        this.comNum = r.nextInt(3);
        return comNum;
    }

    public void setTomGuessNum(int tomNum) {
        this.tomNum = tomNum;
    }

    public int Compare() {
        if ((tomNum == 0 && comNum == 1) || (tomNum == 1 && comNum == 2) || (tomNum == 2 && comNum == 0)) {
            this.winCountNum += 1;
            this.count += 1;
            return 0;
        } else if (tomNum == comNum) {
            return 1;
        } else {
            this.lossCountNum += 1;
            this.count += 1;
            return 2;
        }
    }
}
