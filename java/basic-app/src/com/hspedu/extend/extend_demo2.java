package com.hspedu.extend;

public class extend_demo2 {
    public static void main(String[] args) {
        PC pc = new PC("i3","ssd","8g","lenovo");
        notePad notepad = new notePad("i3","ssd","8g","yellow");
        pc.getInfo();
        notepad.getInfo();
    }
}

class Computer {
    private String cpu;
    private String disk;
    private String memory;

    public Computer(String cpu, String disk, String memory) {
        this.cpu = cpu;
        this.disk = disk;
        this.memory = memory;
    }

    public String getCpu() {
        return cpu;
    }

    public void setCpu(String cpu) {
        this.cpu = cpu;
    }

    public String getDisk() {
        return disk;
    }

    public void setDisk(String disk) {
        this.disk = disk;
    }

    public String getMemory() {
        return memory;
    }

    public void setMemory(String memory) {
        this.memory = memory;
    }

    public String getDetails() {
        return "cpu: " + cpu + " memory: " + memory + " disk: " + disk;
    }
}

class PC extends Computer {
    private String brand;

    public PC(String cpu, String disk, String memory, String brand) {
        super(cpu, disk, memory);
        this.brand = brand;
    }

    public String getBrand() {
        return brand;
    }

    public void setBrand(String brand) {
        this.brand = brand;
    }

    public void getInfo(){
        System.out.print("PC信息为："+getDetails()+" brand: "+brand+"\n");
    }

}

class notePad extends Computer {
    private String color;

    public notePad(String cpu, String disk, String memory, String color) {
        super(cpu, disk, memory);
        this.color = color;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public void getInfo(){
        System.out.print("PC信息为："+getDetails()+" color: "+color+"\n");
    }
}