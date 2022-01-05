package com.hspedu.oopExercise;

public class demo1 {
    public static void main(String[] args) {

        Person[] personArray = new Person[3];
        personArray[0] = new Person("a", 30, "salesclerk");
        personArray[1] = new Person("b", 26, "worker");
        personArray[2] = new Person("c", 35, "manager");

        for (int i = 0; i < personArray.length-1; i++) {
            for (int j = 0; j < personArray.length - i - 1; j++) {
                if (personArray[j].getAge() > personArray[j + 1].getAge()) {
                    Person temp = personArray[j];
                    personArray[j] = personArray[j + 1];
                    personArray[j + 1] = temp;
                }
            }
        }

        for (int i = 0; i < personArray.length; i++) {
            System.out.println(personArray[i]);
        }
    }
}

class Person {
    private String name;
    private int age;
    private String job;

    public Person(String name, int age, String job) {
        this.name = name;
        this.age = age;
        this.job = job;
    }

    @Override
    public String toString() {
        return "Person{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", job='" + job + '\'' +
                '}';
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public String getJob() {
        return job;
    }

    public void setJob(String job) {
        this.job = job;
    }
}