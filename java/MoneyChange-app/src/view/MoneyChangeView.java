package view;

import domain.User;
import tools.Utility;

import java.text.SimpleDateFormat;
import java.util.Date;

public class MoneyChangeView {
    private boolean loop = true;
    private char key = ' ';
    private User user = new User("admin");
    Date date = null;
    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm");
    String details = "===============零钱通明细===============";

    public void detail(){
        //记录零钱通明细
        System.out.println(details);
    }

    public void earn(){
        //完成收益入账功能
        date = new Date();
        System.out.print("请输入入账金额: ");
        int earnMoney = Utility.readInt();
        user.setBalance(user.getBalance()+earnMoney);
        details+="\n"+"收益入账\t+"+earnMoney+"\t"+sdf.format(date)+"\t余额"+ user.getBalance();
        System.out.print("收益入账\t+"+earnMoney+"\t"+sdf.format(date)+"\t余额:"+ user.getBalance());
    }

    public void expense(){
        //实现消费功能
        date = new Date();
        System.out.print("请输入消费金额: ");
        int expenseMoney = Utility.readInt();
        user.setBalance(user.getBalance()-expenseMoney);
        details+="\n"+"消费明细\t-"+expenseMoney+"\t"+sdf.format(date)+"\t余额"+ user.getBalance();
        System.out.print("消费明细\t-"+expenseMoney+"\t"+sdf.format(date)+"\t余额:"+ user.getBalance());
    }

    public void exit(){
        System.out.print("请确认是否退出(y/n):");
        char select = Utility.readConfirmSelection();
        if(select=='Y'){
            loop = false;
        }
    }

    public void mainMenu(){
        do {
            System.out.println("\n===============零钱通菜单===============");
            System.out.println("\t\t1 零钱通明细");
            System.out.println("\t\t2 收益入账");
            System.out.println("\t\t3 消费");
            System.out.println("\t\t4 退出");
            System.out.println("======================================");
            System.out.print("请选择(1~4):");

            char key = Utility.readMenuSelection();
            switch (key) {
                case '1':
                    this.detail();
                    break;
                case '2':
                    this.earn();
                    break;
                case '3':
                    this.expense();
                    break;
                case '4':
                    this.exit();
                    break;
                default:
                    break;
            }
        }while(loop);

    }

}
