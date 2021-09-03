# Linux学习笔记

## 一、基础

### 概述

Linux属于一种**操作系统**，具有很好的稳定性、安全性和处理多并发的能力

**发行版本**包括：Ubuntu/RedHat/CentOS/Debain/Fedora/SuSE/OpenSUSE

Linux最擅长的领域：**服务器**

主要特点：**免费、稳定、高效**，可以在各种语言中广泛应用



### 虚拟机创建

linux分区：boot(1G)、swap(size==内存)、root(剩余)

swap分区作用：可以临时充当内存，但速度不如内存



### 网络连接的三种模式

1、桥接模式：虚拟系统可以和外部系统功能通讯，但是容易产生ip冲突

2、NAT模式：又称网络地址转换模式，虚拟系统可以和外部系统功能通讯，NAT可以将内部网络的私有IP地址转换为公有IP地址，避免ip冲突

3、主机模式：独立的系统



### 虚拟机的克隆、快照、删除和迁移

快照主要用于恢复之前的状态，防止错误操作导致系统异常。

删除：先在vmware移除（并未真正删除），再在资源管理器删除



### linux目录结构

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210707202136777.png" alt="image-20210707202136777" style="zoom:50%;" />

linux下一切皆为文件！

根目录：`/`

子目录：

**/bin：存放经常使用的命令 （包含/usr/bin、/usr/local/bin)**

**/boot：存放linux启动的一些核心文件，包括一些连接文件以及镜像文件**

**/dev：存放Linux 的外部设备，在 Linux 中访问设备的方式和访问文件的方式是相同的。**

**/etc：所有的系统管理所需要的配置文件和子目录**

**/home：存放普通用户的主目录**

**/root：系统管理员的用户主目录**

/tmp：存放一些临时数据

/usr：存放用户的应用程序和文件

/var：存放着在不断扩充着的东西，我们习惯将那些经常被修改的目录放在这个目录下。包括各种日志文件。

/lib：系统开机所需要的最基本的动态连接共享库，类似于Windows的DLL文件。

/sbin：存放系统管理员的系统管理程序

/run：是一个临时文件系统，存储系统启动以来的信息。当系统重启时，这个目录下的文件应该被删掉或清除。如果你的系统上有 /var/run 目录，应该让它指向 run。

/proc：虚拟目录，是系统内存的映射，访问这个目录来获取系统信息（不能动）

/srv：存放一些服务器启动之后需要提取的数据（不能动）

/sys：安装2.6内核的文件系统（不能动）

/media：linux将自动识别的一些设备挂载到这个目录下（u盘、光驱等）

/mnt：允许用户临时挂载别的文件系统，可以通过该文件查看光驱中的内容。

/opt：主机额外安装软件所摆放的目录

/lost+found：系统非法关机后存放的文件



### Linux组

在linux中每个用户必须属于一个组，不能独立于组外。

**所有者：**创建文件的用户

**所在组：**创建文件的用户所在的组

**其他组：**对文件附属组其他的组





### 用户权限（*）

![image-20210903111825574](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210903111825574.png)

当输入`ll`查看文件详细内容时，各位置的意义如下：

（1）第一个位置为文件的类型和权限缩写【共10位】
第0位确定文件类型：-为普通文件、l为链接、d为目录、c为字符设备，如鼠标键盘、b为块设备，如硬盘；
第1-3位为文件所有者对文件的权限；
第4-6位为文件所在组对该文件的权限；
第7-9位表示（组外）其他用户对该文件的权限  

（2）第二个位置为一个数字，如果为文件则为1，如果为目录则显示子目录数

（3）

注意点：
1、rwx分为作用于文件和作用于目录两种，
当**rwx作用于文件时**，r表示可以读取和查看文件，w表示可修改但不可以删除文件，x表示可以执行文件；
当**rwx作用于目录时**，r表示可以ll出目录下拥有哪些文件，w表示可以对目录下文件进行创建、删除和重命名操作，重命名目录；x表示可以进入该目录。





---



## 二、实操

### 创建screen会话(*)

远程服务器的时候，断网或者手误关掉了远程终端，会导致会话中断，程序终止。
而Screen连接的终端，会话独立运行，程序会一直进行。而且会话可以恢复，还可以自行删除。

**常用命令**

```shell
screen -S yourname           # 新建一个叫yourname的session

screen -ls                   # 列出当前所有的session

screen -r yourname           # 回到yourname这个session

screen -d yourname           # 远程detach某个session    # detach快捷键 ctrl a + d

screen -d -r yourname        # 结束当前session并回到yourname这个session

screen -S yourname -X quit   # 删除叫yourname的session

screen -wipe 				 # 清除dead的会话

screen -x					 # 用于会话窗口的共享，比如多台主机连接到该服务器用户端，可以同时共享输出结果
```

**注意点**

* 创建screen会话首先需要切换到ssh会话，然后cd都需要运行的路径。
* 如果遇到无法screen -r的情况，需要首先screen -d
* 如果一个Screen会话中最后一个窗口被关闭了，那么整个Screen会话也就退出了，screen进程会被终止



---



### Vim的使用(*)

vim是vi的增强版，是linux下的文本编辑器，具有程序编辑的能力，可用字体颜色辨别语法的正确性。

vim常用的三种模式：正常模式、插入模式、命令行模式。



**正常模式**

在其他模式下，输入`esc`进入正常模式。可以使用上下左右移动光标，可以使用删除字符和删除整行，也可以使用复制粘贴。

快捷键：

- **输入`u`（小写）撤销刚才的工作**
- **输入`yy`拷贝当前行（`5yy`拷贝当前行向下的5行），`p`粘贴**
- **输入`dd`删除当前行（`5dd`删除当前行向下的5行）**
- `/查找词`+回车，输入n键向下切换 输入N向上切换（重新输入斜杠即可换查找词）
- 输入`:set nu`设置文件的行号，输入`:set nonu`取消文件的行号
- 输入G（大写）定位到文件的最末行，输入`gg`（小写）定位到文件的最首行
- 先输入行号，再输入`shift+g`快速跳转到指定行



**插入模式**

使用i,o,a,r的大小写可以进入，直接进行编辑。



**命令行模式**

输入 : 进入命令行模式。可以完成读取存盘、替换、显示行号和离开等。

命令：

`:wq` ：保存并退出

`:q` ：直接退出

`:q!`  : 强制退出不保存



**补充**

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210713213108951.png" alt="image-20210713213108951" style="zoom:50%;" />





---



### 系统指令

##### 关机和重启

```shell
shutdown -h now  或 init 0	# 立刻关机

shutdown -h 1 或 shutdown	# 1分钟后关机

shutdown -r now 或 init 6 	# 立刻重启（1分钟重启的操作类似上一条）

shutdown -c 		     	 # 取消关机

halt 	 					 # 立刻关机（同上）

reboot 					     # 立刻重启（同上）

sync	 					 # 将内存数据同步到磁盘
```

**注意点：**

* 关机或重启前都需要先运行sync命令，把内存数据写到磁盘(虽然目前的关机重启命令会自动sync，但保险起见最好先同步)

* 这些命令需要在root权限下进行（或在命令前加sudo）






##### 用户管理

```shell
useradd usr			# 添加用户（别忘了添加用户名否则是为当前的用户设置密码）

useradd -d dir usr	# 在指定目录创建用户的主目录

passwd usr			# 为用户设置密码

userdel usr 		# 删除用户保留主目录（一般选择保留）

userdel -r usr		# 删除用户并删除用户主目录

id usr 				# 查看用户id

whoami 				# 查看当前用户（`who am i`	查看用户的开机时间、ip）

su - usr  			# 切换用户（`su - root` 切换到管理员身份）

logout  			# 表示返回原来用户或系统

exit 				# 返回原来用户或关闭终端

groupadd group		 # 新增组

groupdel group		 # 删除组  

useradd -g group usr # 添加用户的同时添加组（注意先写组再写用户）

usermod -g group usr # 修改用户组

usermod -d newpath usr # 修改用户登陆的初始目录（用户需要有进入到新目录的权限）

chgrp group file	 # 修改文件所在的组
```

**注意点**：

* 创建的用户的根目录在root的/home/下

* 在工作中一般不用root用户登陆，而是在普通用户登陆后再用su - 用户名 切换到系统管理员身份

* logout在图形界面无效，运行级别3下有效

* 高权限用户到低权限用户无需输入密码





##### 其他设置

0：关机		1：单用户（找回丢失密码）		2：多用户状态没有网络服务		**3：多用户状态有网络服务（常用）**

4：系统未使用，保留给用户		**5：图形界面（常用）**		6：系统重启

```sh
init [0123456] 					# 修改用户级别

systemctl set-default [...] 	# 指定默认用户级别，如：multi-user.target/graphical.target（centos7之前在/etc/inittab文件中修改）

systemctl get-default 		    # 用于查看默认运行级别

date -s [str]					# 设置时间

tzselect -select a time zone	# 设置时区
```



##### 找回root密码

步骤：

（1）启动系统，进入开机界面，在界面中按“e”进入编辑界面。

（2）进入编辑界面，使用键盘上的上下键把光标往下移动，找到以““Linux16”开头内容所在的行数”，在行的最后面输入：init=/bin/sh。

（3）按快捷键 Ctrl+x 进入**单用户模式**。

（4）接着输入：`mount -o remount,rw /`

（5）接着输入：`passwd` 修改密码并确认

（6）接着输入：`touch /.autorelabel`

（7）最后输入：`exec /sbin/init`，等待系统自动修改密码，完成后，系统会自动重启, 新的密码生效了



---



### 文件目录指令

##### 打印信息

`pwd`	显示当前所在的绝对目录

`ls [-options] path`	打印目录（`-l`以列表形式打印包括权限、所有者的详细信息 `-a`打印包括隐藏文件(.xxx)的信息 `-h`按最大的单位显示大小）[蓝色表示目录 白色表示文件 红色表示压缩包]

`ll path`	查看path目录的详细信息[同ls -l]

`echo -option context`	输出内容到控制台	（可以打印环境变量，如`echo $PATH`)

`history`	查看已经执行过的历史命令	(跟数字表示查看最近x条执行)【历史中可以找到命令的编号，此时**输入`!index`即可重新执行该编号的命令**】

`cal`	显示当月日历信息 （`cal 2020`	显示2020年的所有日历）

`date`	显示当前时间	(`date +%D`	显示年月日	`+%Y`表示年 `+%m` 表示月 `+%d`表示日)	(`date -s “2021-8-16 20:48:00”`设置日期)

`man [命令或配置文件]`	获取帮助信息

`help [命令]` 	获取linux内置命令的帮助信息



##### 查看文件向

`cat file`	仅查看文件内容（类似于vim，但更安全）	

`cat -n file | more` 	该写法被称为管道命令。more指令是基于vi的文本过滤器，以全屏方式按页显示文本文件内容，**[-n]表示显示行号**

more快捷键：**空格（向下一页） enter（向下一行） q（返回，不再显示） =（显示当前行号） :f（输出文件名和行号）**

`cat -n file | less`	less命令和more指令类似，但更强大。在显示文件内容时，根据显示需要加载内容，对显示大型文件有较高效率。

less快捷键：**空格或pagedown（向下一页）pageup(向上一页)  /字符串（向下查询[n向下 N向上]) ?字符串（向上搜寻） q(返回)**

`head file`	默认显示文件的前10行的内容	(`head -n x file`	**[-n]表示选择前x行的内容**)

`tail file`	默认显示文件的后10行内容	（`tail -n x file`	[-n]表示选择后x行的内容 `tail -f file`	**[-f]表示实时监控文件的内容**)



##### 切换目录

`cd path`	切换到指定目录（注意绝对路径前加`/`，相对路径不用加或用`./path`）

`cd /`	切换到主目录，进入系统默认的是用户目录

`cd ~`	或 `cd` 切换到用户目录

`cd ..` 	跳转到上一层（`cd ../../`跳转到上上层）



##### 删除

`rmdir /path/fold`	删除空目录

`rm -rf /path/fold`	删除非空目录【慎重】 -r：递归  -f：强制（不提示）



##### 创建

`touch filename`	创建空文件

`mkdir /path/fold`	创建单级文件	[-p]创建多级目录



##### 拷贝移动

`cp source dest`	拷贝文件到指定目录	[-r]递归复制

`\cp source dest`	强制覆盖复制（不提示）

`mv source dest`	移动文件与目录或重命名【当文件在同一个目录，该命令用于重命名；也可移动并重命名；不需要-r选项】



##### 查找文件或指令

`find [range] -name filename`	按指定文件名查找，如果存在则打印文件路径

`find [range] -user username`	查找属于指定用户名所有文件	

`find [range] -size ±size`	按指定文件大小查找 (+n 大于 -n 小于 n 等于，单位包括k,M,G)

`locate filename`	locate可以快速定位文件路径，查询速度快。但是必须在locate前使用updatedb创建locate数据库（第一次运行需要实时更新）

`which order`	查看指令所在位置

`grep [option] content file`	过滤查找， 一般与管道指令一起使用：如`cat mydata.txt | grep "hello"`或`grep "hello" mydate.txt`【-n 显示行号】



##### 压缩和解压

`gzip file`	压缩文件	

`gunzip file.gz`	解压文件

`zip [option] des_file.zip fi le`	压缩	[-r 递归压缩]	(将/home下的所有文件和子文件夹压缩成myhome.zip：`zip -r /home/myhome.zip /home/`)

`unzip [option] file.zip`	解压	【-d 指定解压后文件的存放目录】 （将myhome.zip解压到/opt/tmp目录下：`zip -d /opt/tmp /home/myhome.zip`）

`tar [option] file.tar.gz`	打包或解压目录	[-c 打包	-x 解包	-v 显示正在处理的文件名	-z 通过gzip支持的压缩或解压缩	-f 指定压缩后的文件名]
压缩：`tar -zcvf file.tar.gz file1 file2`	将file1和file2压缩为file.tar.gz
解压到当前路径：`tar -zxvf file.tar.gz`	(解压到指定路径：`tar -zxvf file.tar.gz -C /des_path`)



##### 其他

`chown user file `	改变文件所有者 

---



### 语法

##### 管道

操作符为`|`，处理由前一个指令传出的正确输出指令到下一个命令

注意点：

1、管道命令只处理前一个命令的正确输出，不处理错误输出；

2、管道命令右边的命令必须能够接收标准输入流命令。

e.g.

`cat -n filename | more`	以全屏形式显示filename的内容



##### 重定向

操作符为`>`和`>>`，`>`表示覆盖写（如果文件不存在则自动创建），`>>`表示追加

语法：

`ls -l > file`	将列表的内容写入到file中（覆盖写）

`ls -al >> file`	将列表的内容（去除.开头的文件）追加到file中（追加）

`cat file1 > file2`	将file1的内容覆盖到file2

`echo context >> file`	在file末尾写入context



##### 软链接

软链接又称为符号链接，类似于windows下的快捷方式，存放着链接其他文件的路径

语法：

`ln -s [origin] [name]`	给原文件创建一个软链接

e.g.

` ln -s /root /home/myroot`	在home目录创建一个软链接并链接到root目录 （此时权限为开头为l）

`rm /home/myroot`	删除home目录下的软链接(注意不要再后面添加/)

































