# Linux学习笔记

# 一、基础

### 概述

Linux属于一种**操作系统**，具有很好的稳定性、安全性和处理多并发的能力

**发行版本**包括：Ubuntu/RedHat/CentOS/Debain/Fedora/SuSE/OpenSUSE

Linux最擅长的领域：**服务器**

主要特点：**免费、稳定、高效**，可以在各种语言中广泛应用



### Linux分区

安装Linux时，默认linux分区：引导分区boot(1G)、交换分区swap(size=2G即可)、根目录/(剩余)

swap分区作用：swap类似于windows的虚拟内存文件，可以临时充当内存，但速度不如内存



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





### 用户权限(*)

![image-20210903111825574](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210903111825574.png)

当输入`ll`查看文件详细内容时，各位置的意义如下：

（1）第一个位置为文件的类型和权限缩写【共10位】
第0位确定文件类型：**-为普通文件、l为链接、d为目录、c为字符设备，如鼠标键盘、b为块设备，如硬盘**；
第1-3位为文件所有者对文件的权限；
第4-6位为文件所在组对该文件的权限；
第7-9位表示（组外）其他用户对该文件的权限  

（2）第二个位置为一个数字，如果该内容显示为1则为一个文件；如果该内容不为1，则为目录，数字表示目录包含的子目录数。

（3）第三个位置和第四个位置分别表示该文件隶属的用户和组。

（4）第五个位置表示文件的大小（字节），如果为文件夹默认显示4096

（5）第六个位置为文件的最后修改时间。

（6）第七个位置为文件名或 链接名->路径



rwx分为作用于文件和作用于目录两种，当**rwx作用于文件时**：
r表示可以读取和查看该文件；
w表示可修改但不可以删除该文件；
x表示可以执行该文件；
当**rwx作用于目录时**：
r表示可以打印出目录下拥有哪些文件（但依然不影响修改目录内的文件）；
w表示可以对目录下文件进行创建、删除和重命名操作，重命名目录；
x表示可以进入该目录（基本要求）。



### 网络

网络环境配置：
（1）自动获取ip（dhcp)：Linux启动后会自动获取IP，但每次获取的ip地址可能不同
（2）手动指定ip（静态）：修改配置文件/etc/sysconfig/network-scripts/ifcfg-ens33



设置主机名方法：
`hostname`	可以查看当前主机名
`vim /etc/hostname`	修改当前主机名
修改后重启生效



设置hosts映射：
`vim /etc/hosts`	修改映射（格式为`ip hostname` ）
`ip hostname`	即可用设置的hostname代替ip



主机名解析过程（hosts和DNS）：
hosts是一个文本文件（lInux下位于/etc下），用于记录IP和hostname的映射关系；DNS是域名系统，是互联网上域名和IP地址相互映射的分布式数据库。一个域名对应一个IP地址，一个IP地址可以对应多个域名，所以多个域名可以同时被解析到一个IP地址。
<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210923212346910.png" alt="image-20210923212346910" style="zoom:50%;" />



**注意点：** 

* 网络需要在同一个网段（ip的前三段）才能进行通讯



### 公钥和私钥

**两者关系**

* 公钥开放在网站或邮件，供大家使用；而私钥是给只自己使用的。
* 用公钥加密的内容只能用私钥解密，用私钥加密的内容只能用公钥解密。
* 公钥的作用主要是加密和验章，私钥的作用主要是解密和签章。



**安全传输**

公钥和私钥是一种不对称的加密方式。如果要实现双方安全的信息发送，需要满足这样的目标：
（1）我发送的内容必须加密，在邮件的传输过程中不能被别人看到。
（2）必须保证是我发送的邮件，不是别人冒充我的。

如果想满足上述的目标，需要收发信息的**双方都拥有公钥和私钥**。

当我发送一个加密的信息给对方时，双方都需要有对方的公钥。
归属性（保证是自己发送的）：我用**我的私钥做签章**，对方就可以用**我的公钥验章**，由于只有我有私钥，可以保证邮件是我发送的。
保密性（信息不被外泄）：我用**对方的公钥加密信息**，对方可以用**对方的私钥解密**，由于只有对方的私钥可以解密，这样可以保证信息不被别人看到，并保证信息在传送过程中没有被修改。



**私钥签章细节**

发送方用Hash函数可以生成信息的**摘要digest**，使用发送方的私钥对digest进行加密（签章）后，可以生成**数字签名signature**。
发送信息时发送方将signature附在信息后面一同发送，当接收方用发送方的公钥解密（验章）signature后，可以得到信息的digest，可以证明**信息是发送方发出的**；
接收方将信息本身使用Hash函数，将结果与上一步得到的digest做对比，如果一致则说明**信息未被修改**。



---



# 二、实操

### 语法

* **管道**

操作符为`|`，处理由前一个指令传出的正确输出指令到下一个命令

注意点：

1、管道命令只处理前一个命令的正确输出，不处理错误输出；

2、管道命令右边的命令必须能够接收标准输入流命令。

e.g.

`cat -n filename | more`	以全屏形式显示filename的内容



* **重定向**

操作符为`>`和`>>`，`>`表示覆盖写（如果文件不存在则自动创建），`>>`表示追加

语法：

`ls -l > file`	将列表的内容写入到file中（覆盖写）

`ls -al >> file`	将列表的内容（去除.开头的文件）追加到file中（追加）

`cat file1 > file2`	将file1的内容覆盖到file2

`echo context >> file`	在file末尾写入context



---

### 创建screen会话(远程专用)

远程服务器的时候，断网或者手误关掉了远程终端，会导致会话中断，程序终止。
而Screen连接的终端，会话独立运行，程序会一直进行。而且会话可以恢复，还可以自行删除。

**常用命令**

`screen -S yourname`	新建一个叫yourname的session

`screen -ls` 	列出当前所有的session

`screen -r yourname`	回到yourname这个session

`screen -d yourname`	远程detach某个session    # detach快捷键 ctrl a + d

`screen -d -r yourname`	结束当前session并回到yourname这个session

`screen -S yourname -X quit`	删除叫yourname的session

`screen -wipe`	清除dead的会话 	

`screen -x`	用于会话窗口的共享，比如多台主机连接到该服务器用户端，可以同时共享输出结果

**注意点**

* 创建screen会话首先需要切换到ssh会话，然后cd都需要运行的路径。
* 如果遇到无法screen -r的情况，需要首先screen -d
* 如果一个Screen会话中最后一个窗口被关闭了，那么整个Screen会话也就退出了，screen进程会被终止



---

### Vim的使用(文本编辑器)

vim是vi的增强版，是linux下的文本编辑器，具有程序编辑的能力，可用字体颜色辨别语法的正确性。

vim常用的三种模式：正常模式、插入模式、命令行模式。



**正常模式**

在其他模式下，输入`esc`进入正常模式。可以使用上下左右移动光标，可以使用删除字符和删除整行，也可以使用复制粘贴。

交互界面：

- **输入`u`（小写）撤销刚才的工作**
- **输入`yy`拷贝当前行（`5yy`拷贝当前行向下的5行），`p`粘贴**
- **输入`dd`删除当前行（`5dd`删除当前行向下的5行）**
- **输入G（大写）定位到文件的最末行，输入`gg`（小写）定位到文件的最首行**
- `/查找词`+回车，输入n键向下切换 输入N向上切换（重新输入斜杠即可换查找词）
- 输入`:set nu`设置文件的行号，输入`:set nonu`取消文件的行号
- 先输入行号，再输入`shift+g`快速跳转到指定行



**插入模式**

使用i,o,a,r的大小写进入插入模式，直接进行编辑。



**命令行模式**

输入 esc进入命令行模式。可以完成读取存盘、替换、显示行号和离开等。

常用命令：

`:wq` ：保存并退出

`:q` ：直接退出

`:q!`  : 强制退出不保存



**补充**

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210713213108951.png" alt="image-20210713213108951" style="zoom:50%;" />





---

### crond的使用(定时任务调度)

任务调度是指系统在某个时间执行特定的命令或程序。crontab是最常用的定时任务软件。我们一般使用`crontab`进行定时任务的设置。

**任务调度指令**

`crontab -e`	进入个人调度进行定时任务编辑

`crontab -l`	查询crontab任务

`crontab -r`	删除当前用户所有的crontab任务

`service crond restart`	重启任务调度



**任务调度编辑**

在个人调度内输入：`*/1 * * * * command`	表示在每个小时的每分钟执行command指令	e.g. `*/1**** ls -l /etc/ > /tmp/to.txt`表示每小时的每分钟将/etc/目录下的内容写入到/tmp/to.txt中

command一般有两种形式，一种是直接的命令，另一种为脚本文件，第二种最为常用；

command为脚本时的操作步骤：

```sh
vim command.sh			# 创建并编辑sh文件
edit(e.g.):
date >> mydate.txt
chmod u+x command.sh	# 赋予sh文件执行权限
crontab -e				# 编辑任务调度
edit(e.g.):
*/1 * * * * command.sh 
```

**注意点**

* 如果有多行则会分别进行任务调度

* 5个占位符分别表示：一个小时的第几分钟、一天中的第几个小时、一个月中的第几天、一年中的第几个月、一周中的星期几【注意用空格隔开】

* 特殊符号的意义： 

  *：表示任何时间，星号/1，表示每..都执行

  ,：表示不连续的时间，如`0 8,12,16 * * * `意为每天的8:00,12:00,16:00各执行一次命令

  -：表示连续的时间范围：如`0 5 * * 1-6`意为每周一到周六的5:00执行命令

  */n：表示每隔多久执行一次：比如`*/10 * * * *`表示每隔10分钟执行一次

* 星期和日期不要同时出现，易出现混淆（如果出现了则在指定的日期和周几均会执行命令）





**应用实例**

每天凌晨2:00将mysql数据库testdb备份到文件中，备份指令：`mysqldump -u root -p密码 数据库 > /home/db.bak`

```sh
crontab -e 
0 2 * * * mysqldump -u root -proot test > /home/db.bak
```



---

### at的使用(单次定时任务调度)

at命令是一次性定时计划任务，at的守护进程atd会在后台模式运行，检查作业队列来运行。执行完一个任务后就不再执行该任务了。

默认情况下，atd守护进程**每60秒**检查作业队列，有作业时，会检查作业运行时间，如果匹配则运行此作业。

**相关指令**

`ps -ef`	查看当前运行的进程【重要】（at命令在使用时，需要保证atd进行的启动），查看是否有atd进程:`ps -ef | grep atd`

`at [option] [time]`	设定at任务

`atq`	查看已有但没有被执行的at任务

`atrm num`	删除编号为num的at任务

**注意点**

* 指定时间的方法：
  1）以hh:mm的方式（小时:分钟）在当天内指定，如果当天已过该时间，则在第二天的该时间执行；
  2）使用midnight,noon,teatime等模糊时间指定；
  3）采用12小时制，在时间后加AM或PM，如12pm；
  4）指定日期和时间（注意日期必须在时间后面），如04:00 2021-03-1；
  5）使用相对计数法，格式为`now + count time-units`，time-units包括hours、minutes、days、weeks，如now + 5 minutes
* 使用ctrl+d结束at命令的输入
* 在设定任务时，backspace会被转化为^H，此时按住ctrl再删除即可



**应用实例**

两天后的下午5点执行`ls /home`:

```sh
at 5pm + 2 days
at> ls /home
input ctrl+d(twice)
```

明天的下午5点将当前时间写入mydate.txt:

```sh
at 5pm tomorrow
at> date > mydate.txt
input ctrl+d(twice)
```



---

### 磁盘分区和挂载

**分区和挂载的原理**

Linux 系统中“一切皆文件”，所有文件都放置在以根目录为树根的树形目录结构中，Linux的多个分区归根结底只有一个根目录，每个分区都只是组成整个文件系统的一部分。

任何硬件设备也都是文件，它们各有自己的一套文件系统（文件目录结构）。当在 Linux 系统中使用这些硬件设备时，只有将Linux本身的文件目录与硬件设备的文件目录合二为一，硬件设备才能为我们所用。合二为一的过程称为“挂载”。
**挂载，指的就是将设备文件中的顶级目录连接到 Linux 根目录下的某一目录（最好是空目录），访问此目录就等同于访问设备文件。**

**注意点**

* 并不是根目录下任何一个目录都可以作为挂载点，由于挂载操作会使得原有目录中文件被隐藏，因此根目录以及系统原有目录都不要作为挂载点，会造成系统异常甚至崩溃，挂载点最好是新建的空目录。
* 如果不挂载，通过Linux系统中的图形界面系统可以查看找到硬件设备，但命令行方式无法找到硬件设备。

**硬盘介绍**

Linux硬盘可以分为IDE硬盘和SCSI硬盘，目前基本使用SCSI硬盘。IDE硬盘，驱动标识符为"hdx~"，SCSI的标识符为"sdx~"。

其中x表示盘号【第几个硬盘】（a为基本盘，b为基本从属盘，c为辅助主盘，d为辅助从属盘），~表示分区（1-5分别对应各个分区）

磁盘分区主要分为基本分区（primary partion）和扩充分区(extension partion)两种，基本分区和扩充分区的数目之和不能大于四个，多于4个的分区称为逻辑分区。

**添加新硬盘并挂载步骤**

1）虚拟机添加新硬盘，选择类型为SCSI
2）对/sdb进行分区，指令为`fdisk /dev/sdb`,设置相关内容
3）格式化硬盘，指令为`mkfs -t ext4 /dev/sdb1`
4）将分区与目录挂载，首先创建一个空目录，使用`mount /dev/sdb1 /newdisk`进行挂载
	（永久挂载）修改/etc/fstab，如下图最后一行所示，编辑新分区信息，可以执行`mount -a`立即生效，也可以重启

![image-20210915204335997](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210915204335997.png)

**相关指令**

`lsblk`	查看设备挂载情况 （[-f]可以查看挂载点）

`fdisk /dev/sdb`	对虚拟机进行分区	（进入后输入命令进行相关操作	n：创建新的分区	d：删除分区	w：写入并退出	q：不保存且退出）

`mkfs -t ext4 /dev/sdb2 `	格式化硬盘（ext4为分区类型）

`mount /dev/sdb1 newfold`	将sdb1挂载到newfold上(该指令在重启后会失效)

`umount /dev/sdb1`或`umount newfold`	卸载

`df -h`	查看系统整体磁盘使用情况

`du -h dir`	查看指定目录的磁盘占用情况	[-h 带计量单位	--max-depth=x 子目录深度	-a 含文件	-c 显示汇总值]	常用：`du -hca --max-depth=1 dir`



---



### 系统指令

* **关机和重启**

`shutdown -h now`  或` init 0`	立刻关机

`shutdown -h 1` 或 `shutdown`	1分钟后关机

`shutdown -r now` 或 `init 6`	立刻重启（1分钟重启的操作类似上一条）

`shutdown -c`	取消关机

`halt`	立刻关机（同上）

`reboot `	立刻重启（同上）

`sync`	将内存数据同步到磁盘

**注意点：**

* 关机或重启前都需要先运行sync命令，把内存数据写到磁盘(虽然目前的关机重启命令会自动sync，但保险起见最好先同步)

* 这些命令需要在root权限下进行（或在命令前加sudo）



* **用户管理**

`su - usr`	切换用户（`su - root` 切换到管理员身份）

`useradd usr`	添加用户

`useradd -d dir usr`	在指定目录创建用户的主目录

`passwd usr`	为用户设置密码(不设置密码，用户不成立)

`userdel usr`	删除用户保留主目录（一般选择保留）

`userdel -r usr`	删除用户并删除用户主目录

`usermod -d newpath usr` 	修改用户登陆的初始目录（用户需要有进入到新目录的权限）

**注意点**：

* 创建的用户的根目录在root的/home/下

* 在工作中一般不用root用户登陆，而是在普通用户登陆后再用su - 用户名 切换到系统管理员身份

* logout在图形界面无效，运行级别3下有效

* 高权限用户到低权限用户无需输入密码



* **组管理**

`groupadd group`	新增组

`groupdel group`	删除组

`useradd -g group usr` 	添加用户的同时添加组（注意先写组再写用户）

`usermod -g group usr` 	修改用户所在组



* **其他设置**

0：关机		1：单用户（找回丢失密码）		2：多用户状态没有网络服务		**3：多用户状态有网络服务（工作常用）**

4：系统未使用，保留给用户		**5：图形界面（常用）**		6：系统重启

`init [0123456]`	修改用户级别

`systemctl set-default [...]`	指定默认用户级别（不能设为0,6），如：multi-user.target/graphical.target（centos7之前在/etc/inittab文件中修改）

`systemctl get-default`	用于查看默认运行级别

`date -s [str]`	设置时间

`tzselect -select a time zone`	设置时区

`service network restart`	重启网络服务



* **找回root密码**

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

##### **打印信息**

`pwd`	显示当前所在的绝对目录

`ls [-options] path`	打印目录（**[-l]以列表形式打印包括权限、所有者的详细信息 [-a]打印包括隐藏文件(.xxx)的信息 [-h]按最大的单位显示大小 [-R] 递归显示**）[蓝色表示目录 白色表示文件 红色表示压缩包]

`ll path`	查看path目录的详细信息(同ls -l)

`echo -option context`	输出内容到控制台	（可以打印环境变量，如`echo $PATH`)

`history`	查看已经执行过的历史命令	(跟数字表示查看最近x条执行)【历史中可以找到命令的编号，此时**输入`!index`即可重新执行该编号的命令**】

`cal`	显示当月日历信息 （`cal 2020`	显示2020年的所有日历）

`date`	显示当前时间	(`date +%D`	显示年月日	`+%Y`表示年 `+%m` 表示月 `+%d`表示日)	(`date -s “2021-8-16 20:48:00”`设置日期)

`id usr`	查看用户id

`whoami`	查看当前用户

`who am i`	查看用户的开机时间、ip

`man [命令或配置文件]`	获取帮助信息

`help [命令]` 	获取linux内置命令的帮助信息



##### **查看文件向**

`cat file`	仅查看文件内容（类似于vim，但更安全）	

`cat -n file | more` 	该写法被称为管道命令。more指令是基于vi的文本过滤器，以全屏方式按页显示文本文件内容，**[-n]表示显示行号**

more交互界面：**空格（向下一页） enter（向下一行） q（返回，不再显示） =（显示当前行号） :f（输出文件名和行号）**

`cat -n file | less`	less命令和more指令类似，但更强大。在显示文件内容时，根据显示需要加载内容，对显示大型文件有较高效率。

less交互界面：**空格或pagedown（向下一页）pageup(向上一页)  /字符串（向下查询[n向下 N向上]) ?字符串（向上搜寻） q(返回)**

`head file`	默认显示文件的前10行的内容	(`head -n x file`	**[-n]表示选择前x行的内容**)

`tail file`	默认显示文件的后10行内容	（`tail -n x file`	[-n]表示选择后x行的内容 `tail -f file`	**[-f]表示实时监控文件的内容**)

`wc file`	查看文件的行数、单词数和字节数	【[-l] 只显示行数	[-w] 只显示字数	[-c] 只显示字节数】
查看目录下所有文件的个数: `ls -l | grep "^-" | wc -l`)





##### **查找文件或指令**

`find [range] -name filename`	按指定文件名查找，如果存在则打印文件路径

`find [range] -user username`	查找属于指定用户名所有文件	

`find [range] -size ±size`	按指定文件大小查找 (+n 大于 -n 小于 n 等于，单位包括k,M,G)

`locate filename`	locate可以快速定位文件路径，查询速度快。但是必须在locate前使用updatedb创建locate数据库（第一次运行需要实时更新）

`which order`	查看指令所在位置

`grep [option] content file`	过滤查找， 一般与管道指令一起使用：如`cat mydata.txt | grep "hello"` 	

`ls -l dir | grep "^-"`	只查询目录下的文件【-n 显示行号	^-表示只查询权限以-开头的文件名（指文件）】





##### **切换目录**

`cd path`	切换到指定目录（注意绝对路径前加`/`，相对路径不用加或用`./path`）

`cd /`	切换到主目录，进入系统默认的是用户目录

`cd ~`	或 `cd` 切换到用户目录

`cd ..` 	跳转到上一层（`cd ../../`跳转到上上层）





##### **删除、创建、移动、拷贝**

`rmdir /path/fold`	删除空目录

`rm -rf /path/fold`	删除非空目录【慎重】 -r：递归  -f：强制（不提示）

`touch filename`	创建空文件

`mkdir /path/fold`	创建单级文件	[-p]创建多级目录

`cp source dest`	拷贝文件到指定目录	[-r]递归复制

`\cp source dest`	强制覆盖复制（不提示）

`mv source dest`	移动文件与目录或重命名【当文件在同一个目录，该命令用于重命名；也可移动并重命名；不需要-r选项】





##### **压缩和解压**

`yum install command`	安装软件（需联网）

`gzip file`	压缩文件	

`gunzip file.gz`	解压文件

`zip [option] des_file.zip file`	压缩	[-r 递归压缩]	(将/home下的所有文件和子文件夹压缩成myhome.zip：`zip -r /home/myhome.zip /home/`)

`unzip [option] file.zip`	解压	【[-d] 指定解压后文件的存放目录】 （将myhome.zip解压到/opt/tmp目录下：`zip -d /opt/tmp /home/myhome.zip`）

`tar [option] file.tar.gz`	打包或解压目录	【[-c] 打包	[-x] 解包	[-v] 显示正在处理的文件名	[-z] 通过gzip支持的压缩或解压缩	[-f] 指定压缩后的文件名]
压缩：`tar -zcvf file.tar.gz file1 file2`	将file1和file2压缩为file.tar.gz
解压到当前路径：`tar -zxvf file.tar.gz`	(解压到指定路径：`tar -zxvf file.tar.gz -C /des_path`)





##### **软链接**

软链接又称为符号链接，类似于windows下的快捷方式，存放着链接其他文件的路径

语法：

`ln -s [origin] [name]`	给原文件创建一个软链接

e.g.

` ln -s /root /home/myroot`	在home目录创建一个软链接并链接到root目录 （此时权限为开头为l）

`rm /home/myroot`	删除home目录下的软链接(注意不要在后面添加/)



##### **查看文件大小**

`du` 显示每个文件和目录的磁盘使用空间

`du -sh`    查看当前所在文件夹的大小	【[-h] 以K M G为单位显示，提高可读性	 [-s] 仅显示目录的总值】

`du -sh *` 查看当前文件夹下各个文件的大小

`du -sh --max-depth=1 /指定文件夹`	查看指定深度的文件夹的大小（深度=1为指定文件夹下的文件）



##### **修改权限**

**修改文件**

`chmod u=rwx,g=rx,o=x file/dir` 	给文件/目录的u、g、o分别赋予权限

`chmod o+w file/dir`	给文件/目录的其他人添加写权限，此时文件会变为绿色（表示文件为可执行文件）

`chmod a-x file/dir`	给文件/目录的所有人删除执行权限，

`chmod 751 file/dir`	相当于`chmod u=rwx,g=rx,o=x file/dir`

**修改所有者**

`chown newowner file/dir`	改变文件所有者 【[-R] 可以对文件夹下的所有内容递归实现修改】

`chown newowner:newgroup file/dir`	改变文件所有者和所在组	

**修改所有组**

`chgrp newgroup file/dir`	修改所在组【[-R] 可以对文件夹下的所有内容递归实现修改】

**注意点**

* 更改权限有两种方式
  方式1：+、-、=变更权限	**u:所有者 g:所有组 o:其他人 a:所有人**(u、g、o的总和)
  方式2：**r=4 w=2 x=1**，利用各位置的加和对权限进行修改
* 进入目录首先需要对文件夹有x的权限



---

### 服务管理

服务本质就是进程，但运行在后台，通常会监听某个端口，等待其他程序的请求，因此又称为守护进程。



**相关指令：**

`netstat - atulnp`	显示所有端口和所有对应的程序

`chkconfig --list [| grep servicename]`	查看服务的各个运行级别是自启动还是关闭

`chkconfig --level x servicename on/off`	修改某服务在某一运行级别的自启动状态

`service servicename [start|stop|restart|reload|status]`	服务管理（开启、暂停、重启、查看状态）

`ls -l /usr/lib/systemd/system | grep servicename`	查看systemctl下服务的信息



`systemctl [start|stop|restart|status|enable|disable|reload] servicename`	服务管理（开启/暂停/重启/查看状态/设置开机启动/设置开机不启动/重新加载配置文件） 

`systemctl list-unit-files [| grep servicename]`	查看服务开机启动状态

`systemctl is-enabled servicename`	查询某个服务是否是自启动的



**firewall指令：**

`firewall-cmd --permanent --add-port=portnumber/protocol`	打开端口(需要重新载入)

`firewall-cmd --permanent --remove-port=portnumber/protocol`	关闭端口(需要重新载入)

`firewall-cmd --reload`	重新载入（reload后打开或关闭端口才生效）

`firewall-cmd --query-port=portnumber/protocol`	查询端口是否开放



**注意点：**

* centos 7.0大部分服务使用systemctl而不是用service
* service指令管理的服务在/etc/init.d查看
* 输入setup可查看所有服务（带*表示服务自启动）
* systemctl指令管理的服务在/usr/lib/systemd/system查看，service指令管理的服务在/etc/init.d中查看
* systemctl打开或者关闭的服务在重启后会恢复默认状态，需要使用enable/disable对某个服务进行自启动开启或关闭



---

### 进程管理(*)

Linux中，每个执行的程序称为一个进程，每个进程都分配一个ID号。每个进程可能以两种方式存在：前台和后台。前台指用户目前屏幕上可以进行操作的进程。后台是实际在操作的进程。

一般系统的服务都是以后台进程的方式存在，通常常驻在系统中，关机才会停止。



**相关指令:**

`ps -aux`	查看当前系统正在执行的进程。(可用grep管道进一步筛选) [`ps -a` 显示当前终端的所有进程信息		`ps -u` 以用户的格式显示进程信息	`ps -x` 显示后台进程运行的参数]

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210924105822260.png" alt="image-20210924105822260" style="zoom: 33%;" />

`ps -ef`	以全格式显示当前所有进程。

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210924113200335.png" alt="image-20210924113200335" style="zoom: 33%;" />

`kill [option] PID `	通过**进程号**杀死/终止进程	[-9 强制停止进程，比如强制关闭正在运行的终端需要加-9]

`killall processname`	杀死**指定名字**的所有进程

`pstree`	显示进程名组成的树	[-p 同时显示PID进程号 -u 同时显示进程用户]





**进程监控：**

`top [-option]`	**动态**显示正在执行的进程。（自动更新）【option：`-d second` 表示每个second秒更新一次进程信息，默认为3s	`-i`使top指令不显示闲置或僵死进程	`-p pid` 通过监控进程ID监控某个进程状态】
top的交互界面：
**P 使进程按CPU使用率排序（默认）** 
**M 按内存使用率排序**	
**N 以PID排序**	
**u+username 查询指定用户的进程情况**	
**k+pid 终止指定进程**	
**q 退出top**

 

**网络端口监控：**

`netstat [-option]`	监控系统网络情况【option: [-an]按一定顺序排列输出 [-p] 显示哪个进程调用】
Proto:使用的网络协议(tcp/udp)	Local Address:本地地址:本地端口（Linux内部的网络地址）	Foreign Address:外部地址:外部端口（不同用户用相同地址、不同端口）



### 软件包管理

**rpm包管理：**

rpm是用于互联网下载包的打包和安装工具，可以生成具有.RPM扩展名的文件。RPM可以通用于Linux的各个发行版本。

rpm包基本格式：firefox-60.2.2-1.el7.centos.x86_64
firefox：包名	60.2.2-1：版本号	el7.centos.x86_64：适用操作系统（i686/i386表示32位系统，noarch表示通用）

**相关指令：**

`rpm -qa`	查看系统中安装所有的rpm包	【[-q] 查询软件包是否安装	[-qi] 包括软件包信息	[-ql] 查询软件包中含有的文件	[-qf filepath] 查看文件所属软件包】



### 远程服务

sshd（secure shell）服务使用ssh协议远程开启其他主机shell的服务。

退出sshd服务：ctrl+d 或 输入logout

**服务状态调整：**

`systemctl status sshd`    	查看服务状态

`systemctl start sshd`           打开服务

`systemctl stop sshd `           	关闭服务

`systemctl restart sshd`            重起服务

`systemctl enable sshd`           设定开机启动

`systemctl disable sshd`          设定开机不启动

`systemctl reload sshd`            重新加载配置文件

`systemctl list-units`                 列出已开启服务当前状态



**具体操作：**

`ssh remoteip`	以当前登录用户名登陆指定ip远程服务器的shell

`ssh remotename@remoteip`	以指定用户名登陆指定ip远程服务器的shell

`scp localfile remotename@remoteip : remotedir`	上传本地文件localfile到远端的remotedir路径

`scp remotename@remoteip:remotefile /localdir`	下载远程文件到本地

`sshd-keygen`	在本地生成密钥（指定保存加密字符的文件（默认直接enter）——-设定密码——-确认密码）



**免密码登陆：**

Linux:

`ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa` 	在客户端生成一对密钥（公钥/私钥）[-t 加密算法类型，这里是使用rsa算法	-P 指定私钥的密码为空（自定义）	-f 指定生成秘钥对保持的位置]

`ssh-copy-id remotename@remoteip`	将客户端公钥发送给服务端(经过ssh-copy-id后接收公钥的服务端会把公钥追加到服务端对应用户的$HOME/.ssh/authorized_keys文件中)

Windows:

`cat id_rsa.pub >> /home/username/.ssh/authorized_keys`	将C://Users//.ssh//id_ras.pub拷贝到服务器的`~/.ssh/authorized_keys`







