# Git 操作笔记

## 一、前置准备：初始化本地仓库并绑定远程

### 1. 安装 Git

前往 https://git-scm.com/downloads 下载安装，安装完成后在终端验证：

```powershell
git --version
```

### 2. 配置用户信息（首次使用时设置一次即可）

```powershell
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
```

### 3. 初始化本地 Git 仓库

在项目根目录下执行：

```powershell
cd "D:\APP\Python 3.13\rag\互联网+"
git init
```

执行后会生成一个隐藏的 `.git` 文件夹，表示该目录已被 Git 管理。

### 4. 创建 .gitignore（排除不需要版本控制的文件）

在项目根目录下创建 `.gitignore` 文件，写入需要排除的规则，例如：

```
__pycache__/
*.pyc
.venv/
.env
configs/setapi.json
data/index/
logs/
*.npy
*.npz
```

这样 `git add .` 时就不会把这些文件纳入版本控制。

### 5. 首次提交

```powershell
git add .
git commit -m "Initial commit"
```

### 6. 在 GitHub 上创建远程仓库

1. 打开 https://github.com/new
2. 填写仓库名称（如 `Internet`）
3. 选择 Public 或 Private
4. 不要勾选"Initialize this repository with a README"（避免远程有初始提交导致冲突）
5. 点击 Create repository

### 7. 绑定远程仓库

```powershell
git remote add origin https://github.com/你的用户名/仓库名.git
```

验证是否绑定成功：

```powershell
git remote -v
```

### 8. 首次推送到远程

```powershell
git push -u origin main
```

`-u` 参数会设置上游跟踪，之后直接 `git push` 即可，不用再指定分支名。

> 如果远程仓库创建时勾选了 README，导致本地和远程历史不一致，可以先执行 `git pull origin main --allow-unrelated-histories` 拉取合并，再 push。

---

## 二、日常开发操作

### 1. 查看当前状态

```powershell
git status          # 查看哪些文件被修改、暂存或未跟踪
git log --oneline   # 查看提交历史（简洁模式）
```

### 2. 暂存与提交

```powershell
git add .                              # 暂存所有更改
git commit -m "提交说明"               # 提交到本地仓库
```

### 3. 推送到远程

```powershell
git push origin main                   # 推送到远程 main 分支
git push origin 430                    # 推送到远程 430 分支
```

### 4. 拉取远程更新

```powershell
git pull origin main                   # 拉取远程 main 分支并合并到当前分支
```

---

## 三、分支操作

### 1. 查看分支

```powershell
git branch        # 本地分支（当前分支前标 * 号）
git branch -r     # 远程分支
git branch -a     # 全部（本地 + 远程）
```

### 2. 创建与切换分支

```powershell
git checkout -b 新分支名        # 创建并切换到新分支
git checkout 分支名             # 切换到已有分支
```

### 3. 合并分支

```powershell
git checkout main               # 先切换到目标分支
git merge 430                   # 将 430 分支的内容合并到 main
```

### 4. 将本地分支推送到远程新分支

```powershell
git push origin 本地分支名:远程分支名
# 例如：将本地 430 推送到远程 51
git push origin 430:51
```

### 5. 删除分支

```powershell
git branch -D 本地分支名        # 删除本地分支（强制）
git push origin --delete 远程分支名   # 删除远程分支
```

---

## 四、完整操作流程示例

以下是从修改代码到推送到多个远程分支的完整流程：

```powershell
# 进入项目目录
cd "D:\APP\Python 3.13\rag\互联网+"

# 暂存所有更改
git add .

# 提交
git commit -m "更新 Web 演示、引导学习与配置"

# 推送到远程 main 分支
git push origin main

# 推送到远程 430 分支
git push origin 430

# 将本地 430 内容推送到远程 51 分支
git push origin 430:51

# 合并 430 到 main
git checkout main
git merge 430

# 删除本地 430 分支
git branch -D 430
```

---

## 五、常用速查

| 操作 | 命令 |
|------|------|
| 初始化仓库 | `git init` |
| 绑定远程 | `git remote add origin <url>` |
| 查看远程 | `git remote -v` |
| 暂存所有 | `git add .` |
| 提交 | `git commit -m "说明"` |
| 推送 | `git push origin <分支名>` |
| 拉取 | `git pull origin <分支名>` |
| 查看分支 | `git branch -a` |
| 新建分支 | `git checkout -b <分支名>` |
| 切换分支 | `git checkout <分支名>` |
| 合并分支 | `git merge <分支名>` |
| 删除本地分支 | `git branch -D <分支名>` |
| 删除远程分支 | `git push origin --delete <分支名>` |
| 查看状态 | `git status` |
| 查看日志 | `git log --oneline` |
