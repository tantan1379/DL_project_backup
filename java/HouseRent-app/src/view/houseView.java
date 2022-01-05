package view;

import domain.House;
import service.houseService;
import utils.Utility;


//界面
public class houseView {
    private boolean loop = true;    //控制显示菜单
    private final houseService h = new houseService(10);

    public void printHouse() {
        House[] houses = h.list();
        System.out.println("===================房屋列表====================");
        if (h.isEmpty()) {
            System.out.println("未找到房屋信息，请先添加房屋...");
            System.out.println("=============================================");
            return;
        }
        System.out.println("编号\t\t房主\t\t电话\t\t地址\t\t月租\t\t是否已出租(y/n)");
        for (House house : houses) {
            if (house == null) {    //如果对象为null则跳过
                break;
            }
            System.out.println(house);  //打印数组中的对象名会调用toString()方法
        }
        System.out.println("================房屋列表显示完毕=================");
    }

    public void addHouse() {
        System.out.println("===================添加房屋====================");
        System.out.print("姓名: ");
        String name = Utility.readString(8);
        System.out.print("电话: ");
        String phone = Utility.readString(12);
        System.out.print("地址: ");
        String address = Utility.readString(16);
        System.out.print("月租: ");
        int rent = Utility.readInt();
        System.out.print("是否已出租(y/n): ");
        String state = Utility.readString(1);
        House newHouse = new House(name, phone, address, rent, state);
        if (h.add(newHouse)) {
            System.out.println("==================添加房屋成功===================");
        } else {
            System.out.println("==================添加房屋失败===================");
        }
    }

    public void deleteHouse() {
        System.out.println("===================删除房屋====================");
        if (h.isEmpty()) {
            System.out.println("未找到房屋信息，请先添加房屋...");
            System.out.println("=============================================");
            return;
        }
        System.out.print("请选择待删除房屋编号(-1退出): ");
        int delId = Utility.readInt();
        if (delId == -1) {
            System.out.println("=================放弃删除房屋==================");
            return;
        }
        System.out.print("您将删除" + delId + "号房屋的信息，请确认是否删除(Y/N):");
        char delChoice = Utility.readConfirmSelection();
        if(delChoice=='Y'){
            if(h.delete(delId)) {
                System.out.println("===================删除完成====================");
            }else{
                System.out.println("您所要删除的房屋不存在...");
            }
        }
        else{
            System.out.println("=================放弃删除房屋==================");
        }
    }

    public void findHouse(){
        System.out.println("===================查找房屋====================");
        if (h.isEmpty()) {
            System.out.println("未找到房屋信息，请先添加房屋...");
            System.out.println("=============================================");
            return;
        }
        System.out.print("请输入您要查找的房屋编号: ");
        int searchId = Utility.readInt();
        House house = h.findById(searchId);
        if(house!=null){
            System.out.println("编号\t\t房主\t\t电话\t\t地址\t\t月租\t\t是否已出租(y/n)");
            System.out.println(house);
        }
        else{
            System.out.println("查找的房屋不存在...");
        }
    }

    public void editHouse(){
        System.out.println("===================修改房屋====================");
        if (h.isEmpty()) {
            System.out.println("未找到房屋信息，请先添加房屋...");
            System.out.println("=============================================");
            return;
        }
        System.out.print("请选择您要修改的房屋编号(-1退出): ");
        int editId = Utility.readInt();
        if (editId == -1) {
            System.out.println("=================放弃修改房屋==================");
            return;
        }
        House house = h.findById(editId);//由于findById()返回的是引用类型（指向堆内存中的制定空间）,因此修改house的属性值会直接影响该对象
        if(house!=null){
            System.out.print("姓名("+house.getName()+"): ");
            String name = Utility.readString(8,"");
            if(!"".equals(name)) {
                house.setName(name);
            }
            System.out.print("电话("+house.getPhoneNum()+"): ");
            String phone = Utility.readString(12,"");
            if(!"".equals(phone)) {
                house.setPhoneNum(phone);
            }
            System.out.print("地址("+house.getAddress()+"): ");
            String address = Utility.readString(16,"");
            if(!"".equals(address)) {
                house.setAddress(address);
            }
            System.out.print("月租("+house.getRent()+"): ");
            int rent = Utility.readInt(-1);
            if(rent!=-1) {
                house.setRent(rent);
            }
            System.out.print("姓名("+house.getState()+"): ");
            String state = Utility.readString(1,"");
            if(!"".equals(state)) {
                house.setState(state);
            }
        }else{
            System.out.println("查找的房屋不存在...");
        }
    }

    public void exit(){
        System.out.print("请确认是否退出(Y/N):");
        char c = Utility.readConfirmSelection();
        if(c=='Y'){
            loop = false;
            System.out.println("=============您已退出，谢谢使用！=============");
        }
    }


    public void mainMenu() {
        do {
            System.out.println("\n==================房屋出租系统==================");
            System.out.println("\t\t\t1 新 增 房 源");
            System.out.println("\t\t\t2 查 找 房 屋");
            System.out.println("\t\t\t3 删 除 房 屋");
            System.out.println("\t\t\t4 修 改 房 屋 信 息");
            System.out.println("\t\t\t5 房 屋 列 表");
            System.out.println("\t\t\t6 退      出");
            System.out.println("=============================================");
            System.out.print("请选择(1~6):");

            //接收用户选择
            char key = Utility.readMenuSelection();
            switch (key) {
                case '1':
                    addHouse();
                    break;
                case '2':
                    findHouse();
                    break;
                case '3':
                    deleteHouse();
                    break;
                case '4':
                    editHouse();
                    break;
                case '5':
                    printHouse();
                    break;
                case '6':
                    exit();
                    break;
                default:
                    break;
            }
        } while (loop);
    }
}

