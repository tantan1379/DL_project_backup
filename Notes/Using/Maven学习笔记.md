# 概述

### **结构（由上往下调用）**

* 浏览器

* 表示层/表述层/表现层（视图层【用于显示界面（H5/CSS/JS/JSP）】-控制层【负责接收请求参数、处理请求、返回响应，跳转页面（Servlet/Action/Handler)】）

* 业务逻辑层

* 持久化层【连接数据库（JDBC/DBUtils/Spring JDBCtemplate/Hibernate/MyBatis）】
* 数据库（MySQL)



### **Maven简介**

Maven 是一个项目管理和构建自动化工具。将项目开发和管理过程抽象成一个项目对象模型（POM：Project Object Model）【一个项目就是一个对象】

它包含了一个项目对象模型（Project Object Model），一组标准集合，一个项目生命周期（Project Lifecycle），一个依赖管理系统（Dependency Management System），和用来运行定义在生命周期阶段（阶段）中插件（插件）目标（目标）的逻辑。


**主要解决问题：**

* jar包版本不匹配，jar包不兼容；
* 工程升级维护过程操作繁琐；

**主要作用：**

* 项目构建：提供标准的、跨平台的自动化项目构建方式
* 依赖管理：方便快捷的管理项目依赖的资源(jar包)，避免资源间的版本冲突问题
* 统一开发结构：提供标准的、统一的项目结构

**具体结构：**pom.xml 

-> 项目对象模型（POM）<-> 依赖管理 -> 本地仓库 -> 私服仓库 -> 中央仓库

![1645175279018](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\1645175279018.png)

**重要概念：**构建生命周期/阶段 <-> 插件



# Maven基础概念

### **仓库**

**用途：**用于存储资源，包括各种jar包。

**分类：**

* 本地仓库：用于存储本地资源的仓库，连接远程仓库获取资源（如果不存在则向私服仓库获取）、
* 远程仓库：非本机电脑的仓库，为本地仓库提供资源
  * 中央仓库：国外Maven团队建立维护的仓库，用于下载和存储jar包。
  * 私服仓库：在部门或公司范围内存储资源的仓库，供本地仓库获取调度，从中央仓库获取资源。

**私服作用：**

* 保护具有版权的资源，包括购买或自主研发的jar（中央仓库的jar包是开源的，不能存储具有版权的资源）
* 一定范围内共享数据（仅对内部开放）



### 坐标

**解释：**Maven中描述仓库中对应资源的位置

**用途：**使用唯一标识，唯一定位资源位置，通过该标识可以将资源的识别与下载工作交由机器完成

**主要组成：**

* groupId：定义当前Maven项目隶属组织名称（通常是域名反写，例如org.mybatis）
* artifactId：定义当前Maven项目名称（通常是模块名称，例如CRM、SMS）
* version：定义当前项目的版本号
* packaging：定义该项目的打包方式

**本地仓库配置**

默认保存位置为：$(user.home)/.m2/repository，需要修改maven下conf中的setting.xml：  `<localRepository>F:\Maven\repository</localRepository>`

**远程仓库配置**

默认访问国外的中央仓库，需要修改为国内阿里的镜像仓库

**全局setting和用户setting**：可以在本地仓库位置外创建一个setting.xml文件，用于实现用户自己的setting（用户覆盖全局）



# Maven指令

### 基本构建命令

`mvn compile`	编译：会同时完成插件下载和编译（将输出文件放在新创建的target中【与src同目录】）；

`mvn clean`	清理：删除编译的target文件；

`mvn test`	测试：对代码进行测试，在测试前会自动编译，在target中生成surefire-reports(测试报告)， test-classes(测试类字节码)两个文件夹；

`mvn package`	打包：打包源程序，会事先自动进行编译、测试，再进行打包（如果没有插件先下载插件）；

`mvn install`	安装到本地仓库：将打包的内容加入到本地库中，会事先自动进行编译、测试、打包；



### 进阶命令



---



# 依赖管理

### 基本实现

依赖可以指定当前项目运行所需的jar包，一个项目可以设置多个依赖

**e.g.**

```xml
<!--设置当前项目所依赖的所有jar-->
<dependencies>
    <!--设置具体的依赖-->
    <dependency>
        <!--依赖所属群组id-->
        <groupId>junit</groupId>
        <!--依赖所属项目id-->
        <artifactId>junit</artifactId>
        <!--依赖版本号-->
        <version>4.11</version>
        <scope>test</scope>
    </dependency>
</dependencies>
```



### 依赖传递

依赖有传递性，可以分为：

直接依赖：在项目中通过依赖配置建立的依赖关系

间接依赖：资源如果依赖其他资源，当前项目间接依赖其他资源



**冲突问题**

路径优先：当依赖中出现相同的资源，层级越深，优先级越低；层级越浅，优先级越高；

声明优先：当资源在相同层级被依赖时，配置顺序靠前的覆盖配置顺序靠后的；

特殊优先：当同级配置了相同资源的不同版本，后配置的覆盖先配置的；



### 可选依赖

可选依赖指对外隐藏当前所依赖的资源：添加`<optional>true<optinal>`



### 排除依赖

排除依赖是指主动断开依赖的资源，被排除的资源无需指定版本

```xml
<!--放在dependencies中-->
<exclusions>
    <exclusion>
    	<groupId>xxx</groupid>
        <artifactId>xxx</artifactId>
    </exclusion>
</exclusions>
```



### 依赖范围

依赖的jar默认情况下可以在任何地方使用，可以通过scope标签设定其作用范围

**作用范围：**

* 主程序范围(main)
* 测试程序范围(test)
* 是否参与打包(package)

**scope类型：**

* compile：在任意范围可以使用（不写时默认该类型）
* test：只在测试程序范围使用
* provided：在主程序和测试程序中使用
* runtime：只在打包中使用



**依赖范围的传递：**只有间接依赖的范围包含package（runtime/compile），才能进行传递



# 生命周期

### **项目构建生命周期**

从项目编译到安装到本地仓库的过程

compile -> test -> test-compile  -> package -> install

Maven对项目构建的生命周期划分为三个部分：clean（清理工作）、default（核心工作，例如编译测试打包部署）、site（产生报告、发布站点）

每个部分又分为若干个阶段（phase）



### **插件**

插件与生命周期内的阶段绑定，**在执行到对应周期时执行对应的插件功能**

默认maven在各个生命周期上绑定有预设功能

通过插件可以自定义其他功能