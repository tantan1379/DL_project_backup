# 概述

### 基础

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



### XML文件

##### **介绍**

xml（Extensible Markup Language）可拓展标记语言。是一种用于标记电子文件使其具有结构性的标记语言。

**特点：**

* XML简单易于在任何应用程序中读/写数据
* xml可用作数据的说明、储存、传输
* xml文件现在多用于作配置文件
* 作为系统与系统之间的传输数据的格式

**xml和html比较：**

* xml文档的标记可以随意扩展，html的标记是预定义的
* xml区分大小写，html不区分大小写
* html主要是用来显示数据的，xml是用来保存数据的
* html中，空格会自动过滤，而xml不会
* html中可以有多个根节点，在xml里面只有一个



##### **语法**

**结构：**一个XML文件通常包含**文件头**和**文件体**两大部分

**文件头（处理指令）**

处理指令主要用于来给处理XML文件的应用程序提供信息的，处理指令的格式：<?处理指令名称 处理指令信息?>

e.g. `<?xml version="1.0" encoding="UTF-8" standalone="yes"？>`

- version：用来表示XML的版本号
- encoding：指定XML编码格式
- standalone：用来表示XML文件是否依赖外部的文件

**文件体**

XML元素定义：`<元素名></元素>`

属性定义：`<元素名 属性="属性值" 属性2="属性值2"></元素>`

注释定义`<!-- 这是一条注释-->`



**转义字符**

\&gt;	大于>

\&lt;	小于<

\&amp;	和号&

\&apos：	单引号‘

\&quot：	引号“



**注意点：**

* xml的标记不能以xml，数字或者下划线开头
* 标签中的属性值要加双引号
* XML**标记**与HTML标记相同，“<”表示一个标记的开始，“>” 表示一个标记的结束。XML中只要有起始标记，就必须有结束标记，而且在使用嵌套结构时，标记之间不能交叉。
* 在XML中不含任何内容的标记叫做**空标记**，格式为：<标记名称/>



##### **结构**

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



### Bean管理（基于xml）

主要负责（1）Spring创建对象；（2）Spring注入属性。

管理操作主要有两种方式：（1）基于**xml配置文件**方式实现；（2）基于**注解方式**实现。



##### **创建对象**

在spring配置文件中，使用bean标签，标签里添加对应属性，即可完成对象的创建。

**语法：**``<bean id="user" class="com.company.User"></bean>``

**注意事项：**

* bean标签有很多属性，常用属性如下：
  * id属性【常用】：给相应对象添加别名，id是bean的唯一标识，IoC容器中bean的id不能重复，否则报错。
  * class属性【常用】：对象所属类的全路径，为bean的全限定类名，指向classpath下类定义所在位置。
  * name属性：name属性基本等同于id属性，不常用，name属性不能重复，且id和name属性也不能重复，和id的区别主要在于name属性可以添加符号。
  * factory-method工厂方法属性：通过该属性，我们可以调用一个指定的静态工厂方法，创建bean实例。
  * factory-bean属性：factory-bean是生成bean的工厂对象，factory-bean属性和factory-method属性一起使用，首先要创建生成bean的工厂类和方法。
* 创建对象时，默认调用类中的无参构造器方法，如果需要使用有参构造器，需要在bean标签中添加properties；



##### **注入属性**

语法：`<property name="xxx" value=“xxx”></property>`	（置于bean标签结构体内）

**知识点：**

* property标签中name表示类中的属性名，value表示向属性注入的值；
* 如果不写bean标签的方法体，则会默认调用无参构造器；
* 如果类有有参构造器，**必须**添加<constructor-arg>标签并添加**所有**属性的值。

**步骤：**

(1) 程序中创建需要实例化的类，定义相关属性；

(2) 在spring配置文件配置对象创建，配置属性注入；

(3) 在测试程序中加载xml文件，获取对象。



**方式一：用set()方法注入属性**

当类中存在set()方法时：

```xml
<bean id="book" class="com.company.Book">
    <!--在xml文件中进行属性注入-->
    <property name="name" value="红楼梦"></property>
</bean>
```



**方式二：通过有参构造器注入属性**

当类中存在有参构造器时：

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

(1) 添加p名称空间在配置文件头部

`xmlns:p="http://www.springframework.org/schema/p"`

(2) 在bean标签进行属性注入（set方式注入的简化操作）

`<bean id="book" class="com.atguigu.spring5.Book" p:bname="very" p:bauthor="good"></bean>`



##### **注入其他类型属性**

(1) 向属性中设置空值：

```xml
<bean id="book" class="com.company.Book">
    <!--在xml文件中进行属性注入-->
    <property name="name" value="红楼梦"></property>
    <property name="price">
        <null/>
    </property>
</bean>
```

(2) 向属性中设置特殊值：

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



##### **外部bean注入**

使用场景：调用**其他包**中的类

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
```
```java
//另一个包中
public class UserDaoImpl implements UserDao {//dao类
    @Override
    public void update() {
        System.out.println("dao update...........");
    }
}
//
public interface UserDao {
	void update();
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



##### **内部bean注入**

使用场景：一个类中包含另一个类的对象

(1) 在同一个包内创建Dept类和Emp类，并生成set方法。

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
        <property name="dep">
            <bean class="com.company.Dep">
                <property name="name" value="microsoft"/><!--内部bean注入-->
            </bean>
        </property>
    </bean>
```

(2.1) 也可以通过级联赋值注入属性（类似于外部注入），在spring配置文件中配置

```xml
    <bean id="emp" class="com.twhupup.Emp">
        <property name="name" value="twh"/>
        <property name="age" value="18"/>
    	<property name="dep" ref="dep"></property>
    </bean>
    <bean id="dep" class="com.twhupup.Dep">
        <property name="name" value="microsoft"></property>
    </bean>
```



##### **注入集合类型属性**

（1）一般注入

注入数组类型属性

 ```xml
<property name="courses">
    <array>
        <value>java课程</value>
        <value>sql课程</value>
        <value>spring课程</value>
    </array>
</property>
 ```

注入List集合类型属性

```xml
<property name="list">
    <list>
        <value>twh</value>
        <value>sjy</value>
        <value>lm</value>
    </list>
</property>
```

注入Map集合类型属性

```xml
<property name="map">
    <map>
        <entry key="JAVA" value="java"/>
        <entry key="PHP" value="php"/>
    </map>
</property>
```

注入Set集合类型属性

```xml
<property name="set">
    <set>
        <value>java</value>
        <value>php</value>
        <value>mysql</value>
    </set>
</property>
```

(2) 在集合中设置对象类型值

```xml
<!--创建多个Course对象-->
<bean id="course1" class="com.twhupup.collectiontype.Course">
    <property name="name" value="Spring5框架"/>
</bean>
<bean id="course2" class="com.twhupup.collectiontype.Course">
    <property name="name" value="Mybatis框架"/>
<bean id="Stu" class="com.twhupup.collectiontype.Stu">
    <!--将创建的对象注入到集合类型中-->    
	<property name="courseList">
        <list>
            <ref bean="course1"/>
            <ref bean="course2"/>
        </list>
	</property>
</bean>
```

(3) 使用util标签提取list集合属性注入

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       <!--在spring配置文件中引入名称空间util-->
       xmlns:util="http://www.springframework.org/schema/util"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
                           http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd">
    <!--1.提取list集合类型属性-->
    <util:list id="bookList">
        <value>java指南</value>
        <value>php指南</value>
        <value>python指南</value>
    </util:list>
    <!--2.list集合类型属性注入-->
    <bean id="book" class="com.twhupup.collectiontype.Book">
        <property name="list" ref="bookList"/>
    </bean>
</beans>
```



### FactoryBean

在Spring中有两种类型的bean，一种为普通bean，另外一种叫做FactoryBean。

普通bean：在bean配置文件中定义的类型返回对应的类型；

工厂bean：在bean配置文件中定义类型和返回类型不一致。



创建工厂bean的步骤：

* 创建类，让这个类作为工厂bean，实现接口FactoryBean；
* 实现接口中的方法，在实现方法中定义返回bean的类型。



### Bean作用域

在 Spring 中默认情况 bean 是单实例对象。需要设置作用域时：

（1）在 spring 配置文件 bean 标签里面有属性（scope）用于设置单实例还是多实例

（2）scope 属性值（默认）singleton，表示是单实例对象；另一种属性值 prototype，表示是多实例对象

```xml
<bean id="book" class="com.atguigu.spring5.collectiontype.Book" scope="prototype"><!--设置为多实例-->
        <property name="list" ref="bookList"></property>
</bean>
```



单实例和多实例的创建时刻：

设置 scope 值是 singleton 时候，**加载 spring 配置文件时**就会创建单实例对象 ；设置 scope 值是 prototype 时候，不是在加载 spring 配置文件时候创建对象，**在调用 getBean 方法时**创建多实例对象




### Bean生命周期

生命周期是对象从创建到销毁的过程，bean生命周期也就是bean对象的创建到销毁过程。

bean的生命周期过程：

（1）通过构造器创建bean对象实例（无参构造）

（2）为bean属性设置值和其它bean引用（调用set方法）

（3）将bean实例传递给bean的后置处理器方法postProcessBeforeInitialization

（4）调用bean的初始化init-method方法（需要进行配置初始化）

（5）将bean实例传递给bean的后置处理器方法postProcessAfterInitialization

（6）得到bean对象，使用bean

（7）容器关闭后，调用bean的销毁方法（需要进行配置销毁）X



### xml自动装配

根据指定的装配规则（属性名称或属性类型），Spring会自动将匹配的属性值进行注入。autowire是在bean标签中设置的。

（1）根据属性名称进行自动装配

在bean标签中添加`autowire="byName"`，再添加需要插入的属性的bean标签

**注意点：**注入值bean的id与需要插入的类属性名称一致，否则注入失败（不报错）

（2）根据属性类型进行自动装配

在bean标签中添加`autowire="byType"`，