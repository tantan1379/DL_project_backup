#### 在vscode离线更新linux上的vscode-server

参考：[visual studio code - How can I install vscode-server in linux offline - Stack Overflow](https://stackoverflow.com/questions/56671520/how-can-i-install-vscode-server-in-linux-offline)

1. First get commit id
2. Download vscode server from url: `https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable`
3. Upload the `vscode-server-linux-x64.tar.gz` to server
4. Unzip the downloaded `vscode-server-linux-x64.tar.gz` to `~/.vscode-server/bin/${commit_id}` without vscode-server-linux-x64 dir
5. Create `0` file under `~/.vscode-server/bin/${commit_id}`

```sh
commit_id=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Download url is: https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable
curl -sSL "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" -o vscode-server-linux-x64.tar.gz

mkdir -p ~/.vscode-server/bin/${commit_id}
# assume that you upload vscode-server-linux-x64.tar.gz to /tmp dir
tar -zxvf /tmp/vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip 1
touch ~/.vscode-server/bin/${commit_id}/0
```



#### FL980键盘快捷键

**Fn快捷键**
+ctrl 调灯光模式
+shift 调颜色
+上下 调亮度
+左右 调速度
+F1 打开音乐播放器
+F2 音量降低
+F3 音量上升
+F4 静音
+F5 暂停 
+F6 下一首
+F7 开始播放
+F8 上一首
+F9 打开浏览器
+F10 打开主页
+F11 打开计算器
+F12 键盘锁



**小键盘**
9：上一页
3：下一页
7：home 移动到当前行最前
1：end 移动到当前行最后
8246分别对应上下左右





#### 计算机网络

OSI 7层参考模型：(未实现)物理层 链路层 网络层 传输控制层 会话层 表示层 应用层



TCP/IP模型： 物理层 链路层 网络层 传输控制层 应用层（三合一）

注意点：前四层由于大体一致因此由**内核**实现，应用层通常追求个性化因此由**程序**实现



四次分手：

1、任一方发送fin

2、对方回复fin+ack（表示收到）

3、对方再次发送fin表示确认分手

4、任一方回复ack表示确认