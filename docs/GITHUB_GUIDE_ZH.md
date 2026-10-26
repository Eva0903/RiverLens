# Windows第一次建立GitHub仓库

本教程采用GitHub Desktop图形界面。这里的“项目”指Repository（代码仓库），暂时不需要另外创建GitHub Projects看板。

## 1. 准备文件

下载发行包并解压。打开`jinjiang-water-quality`文件夹，应直接看到`README.md`、`src`、`experiments`、`configs`和`.gitignore`。

继续保留你原来的`Jinjiang_LSTM_Upgrade_V4`目录；不要把原目录整体拖去上传。本发行包未包含水质数据，也不需要你先运行实验。

## 2. 安装并登录

1. 在[github.com](https://github.com/)注册或登录自己的账号。
2. 从[GitHub Desktop](https://desktop.github.com/)安装Windows客户端。
3. 打开客户端，选择Sign in to GitHub.com，按浏览器提示授权自己的账号。
4. 如果要求设置Git身份，可选用与GitHub账号关联的姓名和邮箱；这会用于代码提交记录。

## 3. 创建一个新的本地仓库

在GitHub Desktop中点击**File → New repository**；初始欢迎页也可能显示**Create a New Repository on your Hard Drive**。

| 字段 | 建议填写 |
|---|---|
| Name | `jinjiang-water-quality` |
| Description | `Indicator-specific information diagnostics for urban river water-quality forecasting` |
| Local path | 选择一个新的父文件夹，例如`C:\Research\GitHub` |
| Initialize this repository with a README | 可不勾选，代码包已经带有README |
| Git ignore | None，使用代码包里的`.gitignore` |
| License | None，暂不替作者决定授权方式 |

点击**Create repository**。它将在所选父文件夹下创建`jinjiang-water-quality`。确认这是一个新的目录，而不是你正在运行实验的原项目。

## 4. 把准备好的文件复制进去

1. 打开刚创建的仓库，例如`C:\Research\GitHub\jinjiang-water-quality`。
2. 把压缩包内`jinjiang-water-quality`文件夹**里面的全部内容**复制进去，包括`.gitignore`、`.gitattributes`及`.github`目录。
3. 如果提示同名README或`.gitattributes`，选择代码包中的版本。
4. 不要再套一层同名文件夹。仓库根目录应直接看到`src`和`README.md`。
5. 不要复制原项目的`data/raw`、`results`或`.venv_gpu`。

## 5. 检查即将提交的文件

回到GitHub Desktop的**Changes**页。应看到源码、配置和文档；`data`下只应有两个README，不应出现Excel、逐样本CSV、模型权重或环境目录。

还可以在仓库根目录打开PowerShell，使用已安装的Python运行：

```powershell
python scripts\audit_public_release.py
```

此命令只用Python标准库，检查文件，不运行模型。若`python`不可用，可尝试`py`或你原GPU环境中Python的完整路径。看到`PASS`后仍简要查看Changes文件列表。

## 6. 做第一次本地提交

在左下方**Summary**填写：

```text
Add research code for experiments 01-07 without observation data
```

点击**Commit to main**（如果默认分支名称不同，按钮也会不同）。这一步只在电脑上记录版本，还没有上传到GitHub。

## 7. 发布到GitHub

1. 点击上方**Publish repository**。
2. 仓库名称保持`jinjiang-water-quality`。
3. **建议保留勾选Keep this code private**，先作为自己的Private仓库；以后决定公开时再调整。
4. 发布到个人账号，Organization选择None（若有该字段）。
5. 点击**Publish Repository**。
6. 完成后点击**Repository → View on GitHub**检查网页。

此时你已经有一个GitHub仓库。网页首页会展示README，文件列表中应包含`src`、`experiments`、`configs`、`tests`和`docs`。

## 8. 之后如何更新

在这个仓库目录内修改代码 → GitHub Desktop查看Changes → 填写本次修改说明 → Commit to main → **Push origin**。例如说明可以写`Clarify meteorological input groups in README`。

不要把一次小修改和大量结果文件混在一起提交。已经在原项目中进行的新代码改动，先复制相应源码到这个仓库，再查看差异。

## 9. 仅供备选：PowerShell命令行上传

如果熟悉Git，可不使用Desktop。在GitHub网页点击“+ → New repository”，创建同名空仓库，选择Private；不要初始化README、`.gitignore`或License。安装[Git for Windows](https://git-scm.com/downloads/win)后，在代码包根目录依次执行：

```powershell
git init -b main
git config user.name "你的GitHub显示名称"
git config user.email "你的GitHub关联邮箱"
python scripts\audit_public_release.py
git add .
python scripts\audit_public_release.py --staged
git diff --cached --name-only
git commit -m "Add experiments 01-07 without observation data"
git remote add origin https://github.com/你的用户名/jinjiang-water-quality.git
git push -u origin main
```

替换用户名和邮箱，不要原样执行占位符。认证按Git凭据管理器弹出的浏览器流程完成；不要把密码或访问令牌写进脚本或远程地址。Desktop方案和命令行方案任选一个即可。

## 常见问题

- **网页看不到内容**：本地Commit后还需要Publish或Push。
- **数据文件没有出现在Changes里**：这是`.gitignore`正常生效。
- **只有ZIP出现在仓库里**：应上传解压后的源码，而不是把整个发行ZIP作为代码仓库。
- **原本已经有同名仓库**：不要强制推送覆盖；先判断是继续原仓库还是改用新名称。
- **需要别人查看Private仓库**：到仓库Settings中添加Collaborator，或之后明确决定公开。仅发送Private仓库网址不会自动授予访问权限。
- **首次Actions检查未通过**：先看失败日志；本地验证记录不表示GitHub云端工作流已经运行成功。

## 官方参考

- [GitHub Desktop创建首个仓库](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop)
- [发布现有项目](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop)
- [命令行上传已有代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

界面字段已按2026年10月26日官方文档核对。
