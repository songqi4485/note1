# 3.Git常用命令

## 3.1设置用户签名

<mark>git config --global user.name "songqi"</mark>

<mark>git config --global user.email 2216969969@qq.com</mark>

* 省略(Local):本地配置，只对本地仓库有效
* --global:全局配置，所有仓库生效
* --system:系统配置，对所有用户生效

说明：

<span style="color: red;">用户签名位置：C:\Users\用户家目录\.gitconfig</span>

## 3.2初始化本地库

<mark>git init</mark>

在初始化之前，在本地创建文件夹，然后右键用git bash打开。执行命令之后会发现文件夹多?git文件?

## 3.3查看本地库状态

<mark>git status</mark>

![img](https://raw.githubusercontent.com/songqi4485/git/main/5e047e3a30a8ab9da0299b5231b8e9b8.png)

<center>检测到未被追踪的文件</center>

## 3.4添加暂存区

添加某个文件<mark>git add README.md</mark>

添加某个目录<mark>git add src/</mark>

添加某个匹配项：<mark>git add *.c</mark>

![img](https://raw.githubusercontent.com/songqi4485/git/main/7a6682969ba32cca7a68d7db5929871a.png)

<center>检测到暂存区有新文件</center>

## 3.5提交到本地库

<mark>git commit -m "日志信息" 文件?</mark>

![img](https://raw.githubusercontent.com/songqi4485/git/main/84432d977f152826024649f25b8fe9c2.png)

<center>提交之后的仓库状态</center>

### 1?*分文件提交与目录结构**

假设文件?`docs/` 里你改了多个文件，希望每个文件单独一条日志：

```bash
# 1) 只暂存一个文?
git add docs/a.md
# 2) 只提交这个文件（本次提交日志只描述它?
git commit -m "docs: update a.md"

git add docs/b.md
git commit -m "docs: revise b.md"

git add docs/c.md
git commit -m "docs: fix typos in c.md"

git push <仓库名> main
```

要点?

- **commit 信息属于这一次提?*，你每次只把一个文?`add` 进暂存区，然?`commit`，自然就“每个文件一条日志”?

说明：

<span style="color: red;">采用该方法，最?push ?GitHub 仓库后，这些文件是否会依然保持正确的目录结构（即都位?`docs/` 文件夹下?/span>

## 3.6修改文件

<span style="color: red;">如果修改已经提交的文件，需要再次将修改的文件添加到暂存?/span>

## 3.7历史版本

### 1）查看历史版?

查看版本信息<mark>git reflog</mark>

查看版本详细信息<mark>git log</mark>

![img](https://raw.githubusercontent.com/songqi4485/git/main/dab02f347c6a4f8b9520deb59b9f8e46.png)

<center>历史版本</center>

### 2）版本穿?

首先查看当前的历史记录，可以看到当前是在 087a1a7 这个版本

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git reflog
087a1a7 (HEAD -> master) HEAD@{0}: commit: my third commit
ca8ded6 HEAD@{1}: commit: my second commit
86366fa HEAD@{2}: commit (initial): my first commit
```

切换?86366fa 版本，也就是我们第一次提交的版本

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git reset --hard 86366fa
HEAD is now at 86366fa my first commit
```

切换完毕之后再查看历史记录，当前成功切换到了 86366fa 版本

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git reflog
86366fa (HEAD -> master) HEAD@{0}: reset: moving to 86366fa
087a1a7 HEAD@{1}: commit: my third commit
ca8ded6 HEAD@{2}: commit: my second commit
86366fa (HEAD -> master) HEAD@{3}: commit (initial): my first commit
```

然后查看文件 hello.txt，发现文件内容已经变?

<span style="color: red;">.git目录下的HEAD文件记录了当前指针指向的分支 </span>

![img](https://raw.githubusercontent.com/songqi4485/git/main/ec9212f87be1460a93307137c1b066e0.png)

### 3）Git切换版本原理

底层其实是移动的 HEAD 指针?

![img](https://raw.githubusercontent.com/songqi4485/git/main/335d97475b969005862e59e421c771be.png)

![](https://raw.githubusercontent.com/songqi4485/git/main/335d97475b969005862e59e421c771be.png)

![](https://raw.githubusercontent.com/songqi4485/git/main/335d97475b969005862e59e421c771be.png)

# 4.Git分支操作

![img](https://raw.githubusercontent.com/songqi4485/git/main/b8c253116a502329f301f8091a2f67bf.png)

## 4.1什么是分支

在版本控制过程中，同时推进多个任务，为每个任务，我们就可以创建每个任务的单独分支?

使用分支意味着程序员可以把自己的工作从开发主线上分离开来，开发自己分支的时候，不会影响主线分支的运行?

对于初学者而言，分支可以简单理解为副本，一个分支就是一个单独的副本?

![img](https://raw.githubusercontent.com/songqi4485/git/main/b17c76ffb0f105ebcb21c53d90c54b0a.png)

## 4.2分支的好?

同时并行推进多个功能开发，提高开发效率?

各个分支在开发过程中，如果某一个分支开发失败，不会对其他分支有任何影响?

## 4.3分支的操?

### 1）查看分?

<mark>git branch -v</mark>

### 2）创建分?

<mark>git branch 分支?</mark>

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git branch hot-fix

Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git branch -v
 hot-fix 087a1a7 my third commit （刚创建的新的分支，并将主分?master 的内容复制了一份）
* master 087a1a7 my third commit
```

### 3）修改分?

在maste 分支上做修改

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ vim hello.txt
```

添加暂存区

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git add hello.txt
```

提交本地?

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git commit -m "my forth commit" hello.txt
[master f363b4c] my forth commit
1 file changed, 1 insertion(+), 1 deletion(-)
```

查看分支

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git branch -v
 hot-fix 087a1a7 my third commit （hot-fix 分支并未做任何改变）
* master f363b4c my forth commit （当?master 分支已更新为最新一次提?
的版本）
```

### 4）切换分?

<mark>git checkout 分支?</mark>

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git checkout hot-fix
Switched to branch 'hot-fix'

--发现当前分支已由 master 改为 hot-fix
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (hot-fix)
$
--查看 hot-fix 分支上的文件内容发现?master 分支上的内容不同
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (hot-fix)
$ cat hello.txt
```

在分支上做修改，然后添加到暂存区、提交到本地?

### 5）合并分?

<mark>git merge 分支?</mark>

本质?*把两个分支的历史整合到一?*，让当前分支（比?`main`）包含另一个分支（比如 `hot-fix`）的改动?

说明：

* 不冲突时：Git 会自动把两边的改动合到一?

?<span style="color: red;">如果两边改的是不同文?不同位置 ?直接合并，`main` 同时拥有两边的变化?/span>

<span style="color: red;"> 如果一边新增文??合并?`main` 也会新增该文件?/span>

* **有冲突时**：不是自动“覆盖”，而是需要你选择/手工编辑冲突内容（决定保留哪边或两边组合）?

### 6）产生冲?

冲突产生的表现：后面状态为 <span style="color: red;">MERGING</span>

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master|MERGING)
$ cat hello.txt
hello git! hello atguigu! 22222222222222
hello git! hello atguigu! 33333333333333
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
<<<<<<< HEAD
hello git! hello atguigu! master test
hello git! hello atguigu!
=======
hello git! hello atguigu!
hello git! hello atguigu! hot-fix test
>>>>>>> hot-fix
```

冲突产生的原因：

?合并分支时，两个分支?span style="color: red;">同一个文件的同一个位?/span>有两套完全不同的修改。Git无法替我们决定使用哪一个。必?span style="color: red;">人为决定</span>新代码内容?

查看状态（检测到有文件有两处修改?

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master|MERGING)
$ git status
On branch master
You have unmerged paths.
 (fix conflicts and run "git commit")
 (use "git merge --abort" to abort the merge)

Unmerged paths:
 (use "git add <file>..." to mark resolution)

 both modified: hello.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

### 7）解决冲?

* 编辑有冲突的文件，删除特殊符号，决定要使用的内容

特殊符号?<<<<<< HEAD 当前分支的代?======= 合并过来的代?>>>>>>> hot-fix

```bash
hello git! hello atguigu! 22222222222222
hello git! hello atguigu! 33333333333333
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu!
hello git! hello atguigu! master test
hello git! hello atguigu! hot-fix test
```

* 添加到暂存区
* 执行提交?span style="color: red;">注意：此时使?git commit 命令时不能带文件?/span>?

## 4.4创建分支和切换分支图?

![img](https://raw.githubusercontent.com/songqi4485/git/main/1d588365d9c8bcb90bf069fc9a19a077.png)

![img](https://raw.githubusercontent.com/songqi4485/git/main/ea33775665ca2b47814713d1a1bd939b.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/ebd0098f6a56932dbc0aeb862bd4c250.png)

master、hot-fix其实都是指向具体版本记录的指针。当前所在的分支，其实是由HEAD决定的。所以创建分支的本质就是多创建一个指针?

HEAD如果指向master，那么我们现在就在master分支上?

HEAD如果指向hotfix，那么我们现在就在hotfix分支上?

所以切换分支的本质就是移动HEAD指针?

<span style="color: red;">Git底层玩的就是两个指针：首先一个HEAD指向分支，如master、hot-fix；master、hot-fix则指向具体版?/span>

# 5.Git团队协作机制

## 5.1团队内协?

![img](https://raw.githubusercontent.com/songqi4485/git/main/c8bb078d7bc465a8e989e587c0b67029.png)

## 5.2跨团队协?

![img](https://raw.githubusercontent.com/songqi4485/git/main/f93c0206b60936448b160a24f672f890.png)

# 6.GitHub操作

## 6.1创建远程仓库

## 6.2远程仓库操作

### 1）创建远程仓库别?

<mark> git remote -v 查看当前所有远程地址别名</mark>

<mark>git remote add 别名 远程地址</mark>

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git remote -v

Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git remote add ori https://github.com/atguiguuyueyue/git-shTest.git

Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git remote -v
ori https://github.com/atguiguuyueyue/git-shTest.git (fetch)
ori https://github.com/atguiguuyueyue/git-shTest.git (push)
```

<span style="color: red;">这个地址是在创建完远程仓库后生成的连接，如图所示红框中</span>

![img](https://raw.githubusercontent.com/songqi4485/git/main/e8f417bc8ffbec5e451fd01bb95b75c1.png)

### 2）推送本地分支到远程仓库

<mark>git push 别名 分支</mark>

```bash
Layne@LAPTOP-Layne MINGW64 /d/Git-Space/SH0720 (master)
$ git push ori master
Logon failed, use ctrl+c to cancel basic credential prompt.
Username for 'https://github.com': atguiguuyueyue
Counting objects: 3, done.
Delta compression using up to 12 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 276 bytes | 276.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/atguiguuyueyue/git-shTest.git
 * [new branch] master -> master
```

### 3）克隆远程仓库到本地

<mark>git clone 远程地址</mark>

<mark>git clone -b <分支? --single-branch 远程地址</mark>

clone 会做如下操作?

1、初始化本地仓库?、拉取代码?、创建别?

# 7.SSH免密登录

?前用 HTTPS 推送时会弹出用户名/登录（或 Token）提示；改用 SSH 协议连接 GitHub**，让 Git 通过你本机的 **SSH 私钥**完成身份认证，从?**push/pull/clone 不再反复登录?

## 7.1进入用户家目?

## 7.2删除旧的 `.ssh` 目录(慎重)

⚠️ 实务提醒（很重要）：

* 不建议你在真实环境随便删 `.ssh`**，因为里面可能有你其他平?服务器的密钥、known_hosts 等?

* 如果只是“新?GitHub 用的 key”，通常直接生成?key（可自定义文件名）更安全。GitHub 官方也建议：如果已有 key，避免覆盖，可用自定义名称?

### 1）生成时?`-f` 指定文件名（最推荐?

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_github
```

## 7.3生成 SSH 密钥对（公钥 + 私钥?

```bash
ssh-keygen -t rsa -C atguiguuyueyue@aliyun.com
```

<span style="color: red;">敲三次回?/span>

生成完成后你会得到：

- 私钥：`~/.ssh/id_rsa`?*千万别泄?*?
- 公钥：`~/.ssh/id_rsa.pub`（要上传?GitHub?

## 7.4查看并复制公钥内?

```bash
cd .ssh
ll -a
cat id_rsa.pub
```

进入 `.ssh` 目录确认文件存在，然后把 `id_rsa.pub` 的整段内容复制出来?

## 7.5?GitHub 网页端添?SSH Key

头像 ?**Settings** ?**SSH and GPG keys** ?**New SSH key**，Title 随意，Key 粘贴刚才复制?`id_rsa.pub` 内容，然?Add?

完成后，GitHub 就“认识”你的这台机器（更准确说：认识与你私钥匹配的公钥）?

![img](https://raw.githubusercontent.com/songqi4485/git/main/202b307d20d6883877be3638745def73.png)

![img](https://raw.githubusercontent.com/songqi4485/git/main/32e164c90f8dc51ba354327811c1aec5.png)