package com.hspedu.polymorphic.parameter;

import java.util.Map;

public class testDemo {
    public static void main(String[] args) {
        Worker w1 = new Worker("Worker1", 3000);
        Worker w2 = new Worker("Worker2", 3400);
        Manager manager1 = new Manager("Manager1", 6000, 2000);
        Manager manager2 = new Manager("Manager2", 6500, 3000);
        testDemo t = new testDemo();
        t.showEmpAnnual(w1);
        t.showEmpAnnual(w2);
        t.showEmpAnnual(manager1);
        t.showEmpAnnual(manager2);
        t.doSomething(w1);
        t.doSomething(w2);
        t.doSomething(manager1);
        t.doSomething(manager2);
    }

    public void showEmpAnnual(Employee e) {
        System.out.println(e.getAnnual());
    }

    public void doSomething(Employee e) {
        if (e instanceof Worker) {
            ((Worker) e).work();
        } else if (e instanceof Manager) {
            ((Manager) e).manage();
        } else {
            System.out.println("错误的类型");
        }
    }
}


