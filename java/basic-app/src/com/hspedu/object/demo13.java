package com.hspedu.object;

public class demo13 {
    public static void main(String[] args) {
        PassObject p = new PassObject();
        Circle c = new Circle();
        p.printArea(c, 5);
    }
}

class Circle {
    double radius;

    public Circle() {
    }

    public Circle(double radius) {
        this.radius = radius;
    }

    public double findArea() {
        return radius * radius * Math.PI;
    }

    public void setRadius(double radius) {
        this.radius = radius;
    }
}

class PassObject {
    public void printArea(Circle c, int times) {
        System.out.println("Radius" + "\t" + "Area");
        for (int i = 1; i <= times; i++) {
            c.setRadius(i);
            System.out.println((double) i + "\t\t" + c.findArea());
        }
    }
}