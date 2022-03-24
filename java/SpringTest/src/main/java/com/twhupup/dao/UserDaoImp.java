package com.company.dao;

/**
 * @Project: spring_demo
 * @Package: com.company.dao
 * @Date: 2022/2/3 21:13
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class UserDaoImp implements UserDao{

    @Override
    public void update() {
        System.out.println("UserDao update()...");
    }
}
