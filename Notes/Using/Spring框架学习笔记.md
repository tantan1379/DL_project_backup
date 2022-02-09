# 概述

Spring是轻量级开源的JavaEE框架，引入jar包很少。提供了功能强大的IOC和Aop（两大核心部分）及Web MVC等功能。

**目的：**用于解决企业应用开发的复杂性。

**IOC:** 控制反转，把创建对象过程交给Spring进行管理。

**Aop:** 面向切面，不修改源代码的前提下进行功能增强。

**特点：**

（1）方便解耦，简化开发；（2）Aop编程支持；（3）方便程序的测试；（4）方便和其它框架进行整合；（5）方便进行事务管理操作；（6）降低Java API的开发和使用难度

**基本jar包：**Beans、Core、Context、Expression



**创建Bean对象过程：**

(1) 在src下创建xml配置文件Spring config，并添加bean标签<bean id="user" class="com.company.User"></bean>

(2) 在测试类中添加测试方法，用`@Test`修饰；

(3) 读取配置文件：`ApplicationContext context = new ClassPathXmlApplicationContext("bean1.xml");`

(4) 创建对象：`User user = context.getBean("user", User.class);`

(5) 调用对象进行代码测试。



---

# XML文件

### 概述

xml文件一般指可扩展标记语言。 是一种用于标记电子文件使其具有结构性的标记语言。

**结构：**一个XML文件通常包含**文件头**和**文件体**两大部分

**处理指令**：处理指令主要用于来给处理XML文件的应用程序提供信息的，处理指令的格式：<?处理指令名称 处理指令信息?>



### **语法**

* 标签中的属性值要加双引号；
* 在HTML中是不区分大小写的，而XML**区分大小写**，包括标记，属性，指令等；
* XML的注释与HTML的**注释**相同，以“<!--”开始，以“-->”结束。
* XML**标记**与HTML标记相同，“<”表示一个标记的开始，“>” 表示一个标记的结束。XML中只要有起始标记，就必须有结束标记，而且在使用嵌套结构时，标记之间不能交叉。
* 在XML中不含任何内容的标记叫做**空标记**，格式为：<标记名称/>





### 转义字符

\&gt;	大于>

\&lt;	小于<

\&amp;	和号&

\&apos：	单引号‘

\&quot：	引号“



### 结构

**文件头**

XML文件头由XML声明与DTD文件类型声明组成。其中DTD文件类型声明是可以缺少的，而XML声明是必须要有的，以使文件符合XML的标准规格。**XML声明必须出现在文档的第一行。**

e.g.: `<?xml version="1.0" encoding="gb2312"?>`

解释：

* “< ?”代表一条指令的开始， “?>”代表一条指令的结束；

* “xml”代表此文件是XML文件；
* “ version="1.0" ”代表此文件用的是XML1.0标准；
* “ encoding="gb2312" ” 代表此文件所用的字符集，默认值为Unicode，如果该文件中要用到中文，就必须将此值设定为gb2312。



**文件体**

文件体中包含的是XML文件的内容，XML元素是XML文件内容的基本单元。从语法讲，一个元素包含一个**起始标记**、一个**结束标记**以及标记之间的**数据内容**。

所有的数据内容都必须在某个标记的开始和结束标记内，而每个标记又必须包含在另一个标记的开始与结束标记内，形成嵌套式的分布，只有最外层的标记不必被其他的标记所包含。最外层的是**根元素(Root)**，又称**文件(Document)元素**，所有的元素都包含在根元素内。

**语法：**<标记名称 属性名1="属性值1" 属性名2=“属性值2" ……>内容</标记名称>	(XML元素与HTML元素的格式基本相同)



---

# IOC容器

主要内容：IOC底层原理、IOC接口（BeanFactory)、IOC具体操作（Bean管理【基于xml、基于注解】）

### 概述

控制反转(Inversion of Control, IOC)，是面向对象编程中的一种设计原则，用来**减低代码之间的耦合度**。通过控制反转，对象在被创建的时候，由一个调控系统内所有对象的外界实体将其所依赖的对象的引用传递给它。

也就是说：**把对象创建和对象之间的调用过程，交给Spring进行管理**。



### 底层原理

**使用技术：**（1）使用设计模式：工厂模式；（2）xml解析；（3）使用反射。

e.g.

```java
class UserFactory{
    public static UserDao getDao(){
        String classValue = "className";
        Class clazz = Class.forName(classValue);//通过反射创建对象
        return (UserDao)clazz.newInstance();
    }
}
```

**基本要点：**

* IOC思想需要基于IOC容器，IOC底层是一个对象工厂；
* Spring提供了IOC容器实现的两种方式：
  * BeanFactory接口：是IOC容器基本实现方式，是Spring内部的使用接口，不提供给开发人员进行使用（加载配置文件时候**不会创建对象**，在获取对象时才会创建对象。）
  * ApplicationContext接口：BeanFactory接口的子接口，提供更多更强大的功能，提供给开发人员使用（加载配置文件时候就会把在配置文件的对象进行创建）【推荐使用，服务器在启动过程中就加载对象】



### Bean管理

主要负责（1）Spring创建对象；（2）Spring注入属性。

管理操作主要有两种方式：（1）基于**xml配置文件**方式实现；（2）基于**注解方式**实现。



#### 基于xml方式

##### 1、创建对象

在spring配置文件中，使用bean标签，标签里添加对应属性，即可完成对象的创建。

**格式：**<bean id="user" class="com.company.User"></bean>

**注意事项：**

* bean标签有很多属性，常用属性如下：
  * id属性【常用】：给相应对象添加别名，id是bean的唯一标识，IoC容器中bean的id不能重复，否则报错。
  * class属性【常用】：对象所属类的全路径，为bean的全限定类名，指向classpath下类定义所在位置。
  * name属性：name属性基本等同于id属性，不常用，name属性不能重复，且id和name属性也不能重复，和id的区别主要在于name属性可以添加符号。
  * factory-method工厂方法属性：通过该属性，我们可以调用一个指定的静态工厂方法，创建bean实例。
  * factory-bean属性：factory-bean是生成bean的工厂对象，factory-bean属性和factory-method属性一起使用，首先要创建生成bean的工厂类和方法。
* 创建对象时，默认调用类中的无参构造器方法，如果需要使用有参构造器，需要在bean标签中添加properties；



##### 2、注入属性

语法：`<property name="xxx" value=“xxx”></property>`

**知识点：**

* property标签（写在bean标签的内部）中name表示类中的属性名，value表示向属性注入的值；
* 如果不写bean标签的方法体，则会默认调用无参构造器；
* 如果类有有参构造器，必须添加<constructor-arg>标签并添加**所有**属性的值。



**方式一：用set()方法注入属性**

**步骤：**

(1) 程序中创建需要实例化的类，定义相关属性和setter方法；

(2) 在spring配置文件配置对象创建，配置setter方法的属性注入；

(3) 在测试程序中加载xml文件，获取对象。

```xml
<bean id="book" class="com.company.Book">
    <!--在xml文件中进行属性注入-->
    <property name="name" value="红楼梦"></property>
</bean>
```



**方式二：通过有参构造器注入属性**

**步骤：**

(1) 创建需要实例化的类，定义相关属性和有参构造器方法；

(2) 在spring配置文件配置对象创建，配置有参构造的属性注入。（注意一行写有参构造器的一个属性）

```xml
<bean id="order" class="com.company.Order">
	<!--name属性也可以替换为index表示索引（如index=0表示有参构造器参数列表中的第一个属性），但不能和已经定义的属性重复-->
    <constructor-arg name="address" value="苏州"></constructor-arg>
    <constructor-arg name="name" value="快递"></constructor-arg>
</bean>
```



**方式三：p名称空间注入（了解）**

使用p名称空间注入，可以简化xml配置方式。（注意p名称空间注入相当于set()方法注入，需要类中定义setter方法）

**步骤：**

（1）添加p名称空间在配置文件头部

`xmlns:p="http://www.springframework.org/schema/p"`

（2）在bean标签进行属性注入（set方式注入的简化操作）

`<bean id="book" class="com.atguigu.spring5.Book" p:bname="very" p:bauthor="good"></bean>`



##### 3、注入其他类型属性

向属性中设置空值：

```xml
<bean id="book" class="com.company.Book">
    <!--在xml文件中进行属性注入-->
    <property name="name" value="红楼梦"></property>
    <property name="price">
        <null/>
    </property>
</bean>
```

向属性中设置特殊值：

```xml
<bean id="book" class="com.company.Book">
    <!--在xml文件中进行属性注入-->
    <property name="name" value="红楼梦"></property>
	<property name="name">
    <!--将带特殊符号的内容写入到CDATA-->
    	<value><![CDATA[<<红楼梦>>]]></value>
	</property>
    <property name="price" value="10"></property>
</bean>
```



##### 4、外部bean注入

使用场景：调用其他包中的类

**步骤：**

(1) 创建service类和dao类，在service中调用dao的方法。

```java
public class UserService {//service类
    //创建UserDao类型属性，生成set方法
    private UserDao userDao;
    public void setUserDao(UserDao userDao) {
        this.userDao = userDao;
    }

    public void add() {
        System.out.println("service add...............");
        userDao.update();//调用dao方法
    }
}
public class UserDaoImpl implements UserDao {//dao类
    @Override
    public void update() {
        System.out.println("dao update...........");
    }
}
```
(2) 在spring配置文件中进行配置

```xml
<!--1 service和dao对象创建-->
<bean id="userService" class="com.atguigu.spring5.service.UserService">
    <!--注入userDao对象
        name属性：类里面属性名称
        ref属性：创建userDao对象bean标签id值
    -->
    <property name="userDao" ref="userDaoImpl"></property>
</bean>
<bean id="userDaoImpl" class="com.atguigu.spring5.dao.UserDaoImpl"></bean>
```



##### 5、内部bean注入

**步骤：**

(1) 在实体类之间表示一对多关系，员工表示所属部门，使用对象类型属性进行表示；

```java
//部门类
public class Dept {
    private String dname;
    public void setDname(String dname) {
        this.dname = dname;
    }
}

//员工类
public class Emp {
    private String ename;
    private String gender;
    //员工属于某一个部门，使用对象形式表示
    private Dept dept;
    
    public void setDept(Dept dept) {
        this.dept = dept;
    }
    public void setEname(String ename) {
        this.ename = ename;
    }
    public void setGender(String gender) {
        this.gender = gender;
    }
}
```

(2) 在spring配置文件中配置

```xml
    <bean id="emp" class="com.company.Emp">
        <property name="name" value="twh"/>
        <property name="age" value="18"/>
        <property name="dep"><!--设置对象属性-->
            <bean class="com.company.Dep">
                <property name="name" value="microsoft"/><!--内部bean注入-->
            </bean>
        </property>
    </bean>
```



##### **6、通过级联赋值注入属性**
