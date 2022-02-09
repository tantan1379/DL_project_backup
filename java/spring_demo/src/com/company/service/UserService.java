package com.company.service;

import com.company.dao.UserDao;

/**
 * @Project: spring_demo
 * @Package: com.company.service
 * @Date: 2022/2/3 21:13
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class UserService {
    private UserDao userDao;

    public void setUserDao(UserDao userDao) {
        this.userDao = userDao;
    }

    public void add(){
        System.out.println("UserService add()...");
        userDao.update();
    }
}
