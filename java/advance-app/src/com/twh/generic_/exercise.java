package com.twh.generic_;

import java.util.ArrayList;
import java.util.Comparator;

/**
 * @Project: advance-app
 * @Package: com.twh.generic_
 * @Date: 2022/1/22 14:21
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class exercise {
    public static void main(String[] args) {
        ArrayList<Employee> employees = new ArrayList<>();
        employees.add(new Employee("twh", 10000, new Mydate(11, 3, 1997)));
        employees.add(new Employee("twh", 10000, new Mydate(11, 6, 1997)));
        employees.add(new Employee("twh", 10000, new Mydate(11, 4, 1997)));
        employees.add(new Employee("xca", 8000, new Mydate(5, 2, 1995)));
        employees.add(new Employee("sjy", 9000, new Mydate(2, 1, 1998)));

        employees.sort(new Comparator<Employee>() {
            @Override
            public int compare(Employee o1, Employee o2) {
                int i = o1.getName().compareTo(o2.getName());
                if (i != 0) {
                    return i;
                }
                return o1.getBirthday().compareTo(o2.getBirthday());
            }
        });
        System.out.println("employees" + employees);
    }
}


class Employee {
    private String name;
    private double sal;
    private Mydate birthday;

    public Employee(String name, double sal, Mydate birthday) {
        this.name = name;
        this.sal = sal;
        this.birthday = birthday;
    }

    public String getName() {
        return name;
    }

    public Mydate getBirthday() {
        return birthday;
    }

    @Override
    public String toString() {
        return "员工" +
                "姓名：'" + name + '\'' +
                ", 工资：" + sal +
                ", 生日：" + birthday;
    }
}

class Mydate implements Comparable<Mydate> {
    private int month;
    private int day;
    private int year;

    public Mydate(int month, int day, int year) {
        this.month = month;
        this.day = day;
        this.year = year;
    }

    public int getMonth() {
        return month;
    }

    public int getDay() {
        return day;
    }

    public int getYear() {
        return year;
    }

    @Override
    public String toString() {
        return getYear() + "/" + getMonth() + "/" + getDay();
    }

    @Override
    public int compareTo(Mydate o) {
        int y = year - o.getYear();
        if (y != 0) {
            return y;
        }
        int m = month - o.getMonth();
        if (m != 0) {
            return m;
        }
        return day - o.getDay();
    }
}