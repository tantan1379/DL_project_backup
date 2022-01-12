package com.twh.innerClass_;

public class AnnoymousInnerClassdemo1 {
    public static void main(String[] args) {
        CellPhone cellPhone = new CellPhone();
        cellPhone.alarmClock(new Bell(){
            @Override
            public void ring() {
                System.out.println("小伙伴上课了");
            }
        });
        //相当于Bell bell = new AnnoymousInnerClassdemo1$1();向下转型，此时在cellPhone调用的alarmClock会与运行类型下的该方法动态绑定
        cellPhone.alarmClock(new Bell(){
            @Override
            public void ring() {
                System.out.println("懒猪起床了");
            }
        });
    }
}

interface Bell{
    void ring();
}

class CellPhone{
    public void alarmClock(Bell bell){
        System.out.println(bell.getClass());
        bell.ring();//动态绑定
    }

}