package service;

import domain.House;

/**
 * HouseService.java<=>类 [业务层]
 * //定义House[] ,保存House对象
 * 1. 响应HouseView的调用
 * 2. 完成对房屋信息的各种操作(增删改查c[create]r[read]u[update]d[delete])
 */

//业务层
public class houseService {
    private final House[] houses; //用对象存储数组信息
    private int houseNum = 0;   //记录当前有多少个房屋信息（默认为一个）
    private int idCount = 0;

    public houseService(int size) {//构造器，用于创建对象数组
        houses = new House[size];
    }

    public boolean isEmpty() {
        return houseNum == 0;
    }

    //返回房屋信息
    public House[] list() {
        return houses;//返回实时的数组
    }

    //添加新对象，返回boolean值
    public boolean add(House newHouse) {
        if (houseNum == houses.length) {
            System.out.println("数组已满，不能再添加...");
            return false;
        }
        newHouse.setId(++idCount);
        houses[(++houseNum) - 1] = newHouse;
        return true;
    }

    public boolean delete(int delNum) {
        int index = -1;
        //房屋编号和房屋的下标不一定一样
        for (int i = 0; i < houseNum; i++) {
            if (delNum == houses[i].getId()) {//需要先找到需要删除的房屋编号(此处要使用getId()获取实时的id号)
                index = i;
            }
        }
        if (index == -1) {          //没找到则返回false
            System.out.println(index);
            return false;
        }
        for (int i = index; i < houseNum - 1; i++) {
            houses[i] = houses[i + 1];
//            houses[i].setId(houses[i+1].getId()-1);
        }
        houses[houseNum - 1] = null;  //将对象数组的最后一个元素置空
        houseNum--;                 //房屋数-1
        return true;                //找到返回true
    }

    public House findById(int searchNum) {
        for (int i = 0; i < houseNum; i++) {
            if (searchNum == houses[i].getId()) {//需要先找到需要删除的房屋编号(此处要使用getId()获取实时的id号)
                return houses[i];
            }
        }
        return null;
    }
}
