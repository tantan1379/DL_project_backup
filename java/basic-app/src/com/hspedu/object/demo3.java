package com.hspedu.object;

public class demo3 {
    public static void main(String[] args) {
        Book b = new Book("abc", 200.3);
        b.info();
        b.updatePrice();
        b.info();
    }
}

class Book {
    String name;
    double price;

    public Book(String name, double price) {
        this.name = name;
        this.price = price;
    }

    public void updatePrice() {
        if (price > 150) {
            price = 150;
        } else if (price > 100) {
            price = 100;
        }
    }

    public void info() {
        System.out.println("书名：" + name + " 价格：" + price);
    }
}
