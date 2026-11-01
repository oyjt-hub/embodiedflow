# 推送到 GitHub

目标：`oyjt-hub/embodiedflow`（公开仓库）

## 前置：给 oyjt-hub 账号授权

当前服务器上的 SSH key 属于 **oyjt-ahu** 账号，没有 oyjt-hub 的权限。二选一：

**A. 把服务器公钥加到 oyjt-hub 账号**

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHcYSmDJs5tmPTVjmYTH4MgQVZf7KghmqJDbHzuiZ8YJ 1729731216@qq.com
```

GitHub → Settings → SSH and GPG keys → New SSH key，粘贴上面整行。

**B. 用 Personal Access Token（HTTPS）**

```bash
git remote set-url origin https://<TOKEN>@github.com/oyjt-hub/embodiedflow.git
```

## 然后：建空仓库并推送

1. 在 GitHub 上新建**空**仓库 `embodiedflow`（不要勾选 README / .gitignore / license）
2. 推送：

```bash
cd ~/workspace/embodiedflow
git push -u origin main
```

## 校验

```bash
git ls-remote origin | head -3
curl -s https://api.github.com/repos/oyjt-hub/embodiedflow | head -c 200
```
