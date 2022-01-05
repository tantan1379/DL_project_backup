package com.hspedu.polymorphic.array;

public class demo {
    public static void main(String[] args) {
        Person[] persons = new Person[5];//对象数组
        persons[0] = new Person("A", 18);
        persons[1] = new Teacher("B",22,30000);
        persons[2] = new Teacher("C",23,20000);
        persons[3] = new Student("D",24,100);
        persons[4] = new Student("E",25,97);

        for (int i = 0; i < persons.length; i++) {
            System.out.println(persons[i].say());

            if(persons[i] instanceof Student){
//                Student stu = (Student)persons[i];
//                persons[i].study();
            }
            else if(persons[i] instanceof Teacher){
//                Teacher tea = (Teacher)persons[i];
//                persons[i].teach();
            }
        }
    }
}

