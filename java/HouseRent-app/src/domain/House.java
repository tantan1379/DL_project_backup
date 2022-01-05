package domain;

//model/domain
public class House {
    private int id = 0;
    private String name;
    private String phoneNum;
    private String address;
    private int rent;
    private String state;

    public House(){

    }

    public House(String name, String phoneNum, String address, int rent, String state) {
        this.name = name;
        this.phoneNum = phoneNum;
        this.address = address;
        this.rent = rent;
        this.state = state;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPhoneNum() {
        return phoneNum;
    }

    public void setPhoneNum(String phoneNum) {
        this.phoneNum = phoneNum;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public int getRent() {
        return rent;
    }

    public void setRent(int rent) {
        this.rent = rent;
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    @Override
    public String toString() {
        return id+"\t\t"+name+"\t\t"+phoneNum+"\t\t"+address+"\t\t"+rent+"\t"+state;
    }
}
