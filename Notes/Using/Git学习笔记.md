**Reference:**

https://blog.csdn.net/weixin_30699831/article/details/101982286

https://blog.csdn.net/halaoda/article/details/78661334

---------------------------------------------------------------
## 基础知识

GIT包括三个区域：**工作区、暂存区（缓存区）、本地仓库**，在远端（remote）包括：远程库

工作区(Work directory)：一般就是我们项目的根目录，我们在工作区修改增加代码；完成编辑后，我们用git add 将工作区文件添加到暂存区

暂存区(Stage/Index)：版本库中设立一个暂存区(Stage/Index），作为用来直接跟工作区的文件进行交互，工作区文件的提交(commit)或者回滚(reset)都是通过暂存区，而版本库中除了暂存区之外，文件的提交的最终存储位置是分支(Branch)，在创建版本库的时候默认都会有一个主分支(Master)。

本地仓库(Repository)：我们在为项目添加本地库之后，会在工作区生成一个隐藏目录“.git”，.git目录即为当前工作区的本地版本库



#### **版本控制流程：**

1、修改本地已被跟踪文件(指旧文件)，文件进入未暂存区域
2、将未暂存区域的文件添加到暂存区：`git add files`
3、将暂存区的文件提交给HEAD：`git commit -m ‘commits’`
4、添加新的远端服务器（如果已经存在则跳过此步骤）：`git remote add origin <git_repository_SSH>` 
5、将本地仓库的HEAD推送到远端仓库：`git push -u origin master`
6、更新本地仓库至最新改动：`git pull`



## 基础命令

#### 各类查看

查看各次提交的ID号（操作类似于linux的less）：`git log`





#### 分支控制

查看本地分支：`git branch`

查看本地分支和远程分支：`git branch -a`

创建分支：`git branch mybranch`

切换分支：`git checkout mybranch` 

创建并切换分支：`git checkout -b mybranch`

删除分支：`git branch -d mybranch`



#### 各类还原

取消add（回退到上一次操作）：`git reset HEAD` 

回退到某一版本（取消commit）：`git reset [--soft | --mixed | --hard] [HEAD]`

HEAD后跟^的数量表示回退的版本数，不加时用于取消暂存(add)的文件，直接返回当前的commit版本；--mixed为默认参数，可以省略，有此参数时会重置暂存区的文件与上一次的commit一致; 使用--hard参数时，会撤销工作区所有未提交的修改内容，并**将暂存区和工作区都返回到上一版本**，并删除指定版本到当前所有的commit信息。



让文件回到最近一次commit或add的状态：`git checkout -- file`

先从缓存区中拉取版本还原，如果缓存区为空则到版本库中拉取还原，将工作区的文件进行替换；这里--的作用是为了防止与切换分支的指令冲突。





#### 服务器端相关

显示已有服务器：`git remote`显示服务器端的地址：`git remote -v`

添加服务器：`git add remote [name] git_repository_SSH` 

删除远程服务器：`git remote rm [name]`

修改远程服务器名字：`git remote rename [name_0] [name_1]`



#### 其他

精简显示文件状态：`git status -s` 
tips:A表示新添加到暂存区的文件，M表示已修改，??表示未跟踪，靠左侧表示暂存区，靠右侧表示工作区





## 常用业务

利用线上仓库重建本地仓库

```bash
$ cd ..
$ rm -r git_repository
$ git clone <git_repository_SSH>
$ cd git_repository
```



commit之后仍需修改源代码

```bash
$ git add .
$ git commit --amend # 将add的代码与commit合并
```



回到最初的起点

```bash
$ rm -rf .git
$ git init
$ git add .
$ git commit -m "first commit"
$ git remote add origin <git_repository_SSH>
$ git push -f -u origin master
```



开发分支（dev）合并到 master 分支

```bash
$ git checkout -b dev # 切换到开发分支
$ git pull 
$ git checkout master
$ git merge dev
$ git push -u origin master
```

