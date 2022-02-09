package com.company.testdemo;

import com.company.Order;
import com.company.User;
import com.company.Book;
import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

/**
 * @Project: spring_demo
 * @Package: com.company.testdemo
 * @Date: 2022/1/31 14:17
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestSpring5 {
    @Test
    public void testAdd(){
        //加载spring配置文件
        ApplicationContext context = new ClassPathXmlApplicationContext("bean1.xml");
        //获取配置文件创建对象
        User user = context.getBean("user", User.class);
        System.out.println(user);
        user.add();
    }

    @Test
    public void testBook(){
        ApplicationContext context = new ClassPathXmlApplicationContext("bean1.xml");
        Book book = context.getBean("book",Book.class);
        System.out.println(book);
    }

    @Test
    public void testOrder(){
    ApplicationContext context = new ClassPathXmlApplicationContext("bean1.xml");
    Order order = context.getBean("order", Order.class);
    System.out.println(order);
    }


}
