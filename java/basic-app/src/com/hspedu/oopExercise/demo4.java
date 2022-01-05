package com.hspedu.oopExercise;

public class demo4 {
    public static void main(String[] args) {
        schoolPerson t1 = new Teacher("twh", "boy", 23, 10);
        schoolPerson t2 = new Teacher("xca", "boy", 24, 12);
        schoolPerson s1 = new Student("sjy", "girl", 22, "20205228021");
        schoolPerson s2 = new Student("lm", "boy", 21, "20205228019");
        schoolPerson[] array = {t1,t2,s1,s2};
        Sort s = new Sort();
        s.quickSort(array,0,array.length-1);
        for (int i = 0; i < array.length; i++) {
            if(array[i] instanceof Student) {
                ((Student)array[i]).printInfo();
            }
            if(array[i] instanceof Teacher) {
                ((Teacher)array[i]).printInfo();
            }
        }
        System.out.println("================");
        for (int i = 0; i < array.length; i++) {
            T.test(array[i]);
        }
    }
}

class Student extends schoolPerson{
    private String stu_id;

    public Student(String name, String sex, int age, String stu_id) {
        super(name, sex, age);
        this.stu_id = stu_id;
    }

    public String getStu_id() {
        return stu_id;
    }

    public void setStu_id(String stu_id) {
        this.stu_id = stu_id;
    }

    public void study(){
        System.out.println(getName()+"好好学习");
    }

    public void play(){
        System.out.println(getName()+"爱玩足球");
    }

    public void printInfo(){
        System.out.println("--------------");
        System.out.println("学生的信息：");
        System.out.println(super.basicInfo());
        System.out.println("学号: "+stu_id);
        study();
        play();
    }

    @Override
    public String toString() {
        return "Student{" +
                "stu_id='" + stu_id + '\'' +
                '}';
    }
}

class Teacher extends schoolPerson{
    private int work_age;

    public Teacher(String name, String sex, int age, int work_age) {
        super(name, sex, age);
        this.work_age = work_age;
    }

    public int getWork_age() {
        return work_age;
    }

    public void setWork_age(int work_age) {
        this.work_age = work_age;
    }

    public void teach(){
        System.out.println(getName()+"认真教课");
    }

    public void play(){
        System.out.println(getName()+"爱玩篮球");
    }

    public void printInfo(){
        System.out.println("--------------");
        System.out.println("老师的信息：");
        System.out.println(super.basicInfo());
        System.out.println("工龄: "+work_age);
        teach();
        play();
    }

    @Override
    public String toString() {
        return "Teacher{" +
                "work_age=" + work_age +
                '}';
    }
}



class schoolPerson{
    private String name;
    private String sex;
    private int age;

    public schoolPerson(String name, String sex, int age) {
        this.name = name;
        this.sex = sex;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public void play(){

    }

    public String basicInfo(){
        return "姓名: " + name + "\n年龄: " + age + "\n性别: " + sex;
    }


    @Override
    public String toString() {
        return "schoolPerson{" +
                "name='" + name + '\'' +
                ", sex='" + sex + '\'' +
                ", age=" + age +
                '}';
    }
}

class Sort{
    public void quickSort(schoolPerson[] array, int L, int R){
        int left = L;
        int right = R;
        int pivot = array[left].getAge();
        schoolPerson pivotObject = array[left];
        if(left>=right){
            return;
        }

        while(left<right){
            while(left<right && array[right].getAge()>=pivot){
                right--;
            }
            array[left] = array[right];
            while(left<right && array[left].getAge()<=pivot){
                left++;
            }
            array[right] = array[left];
        }
        array[left]=pivotObject;
        quickSort(array,L,left-1);
        quickSort(array,right+1,R);
    }
}

class T{
    public static void test(schoolPerson p) {
        if(p instanceof Student) {//p 的运行类型如果是Student
            ((Student) p).study();
        } else if(p instanceof  Teacher) {
            ((Teacher) p).teach();
        } else {
            System.out.println("do nothing...");
        }
    }
}