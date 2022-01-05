package com.hspedu.oopExercise;

import java.util.Objects;

public class demo3 {
    public static void main(String[] args) {
        SavingsAccount s = new SavingsAccount(1000);
        BankAccount b = new BankAccount(1000);
        System.out.println(b.equals(s));
//        s.deposit(100);
//        s.deposit(100);
//        s.deposit(100);
//        System.out.println(s.getBalance());
//        s.withdraw(100);
//        s.withdraw(100);
//        s.withdraw(100);
//        System.out.println(s.getBalance());
//        s.earnMonthlyInterest();
//        System.out.println(s.getBalance());
//        s.deposit(100);
//        s.deposit(100);
//        System.out.println(s.getBalance());
//        s.withdraw(100);
//        System.out.println(s.getBalance());
//        s.withdraw(100);
//        System.out.println(s.getBalance());

    }
}

class SavingsAccount extends BankAccount {
    private double rate = 0.01;
    private int count = 3;

    public void earnMonthlyInterest(){
        count = 3;
        super.deposit(getBalance()*rate);
    }

    public SavingsAccount(double initialBalance) {
        super(initialBalance);
    }

    @Override
    public void deposit(double amount) {
        if (count > 0) {
            super.deposit(amount);
        } else {
            super.deposit(amount-1);
        }
        count--;
    }

    @Override
    public void withdraw(double amount) {
        if (count > 0) {
            super.withdraw(amount);
        } else {
            super.withdraw(amount+1);
        }
        count--;
    }

}


class BankAccount {//父类
    private double balance;//余额

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    //存款
    public void deposit(double amount) {
        balance += amount;
    }

    //取款
    public void withdraw(double amount) {
        balance -= amount;
    }

    public double getBalance() {
        return balance;
    }

    public void setBalance(double balance) {
        this.balance = balance;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass())
        {
            System.out.println("yes");
            return false;
        }
        BankAccount that = (BankAccount) o;
        return Double.compare(that.getBalance(), getBalance()) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(getBalance());
    }
}