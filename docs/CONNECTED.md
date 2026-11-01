# 服务器与 GitHub 的连接

本仓库由 lc 服务器（oyjt 账号）通过专用 SSH key 推送到 GitHub `oyjt-hub`。

- key: `~/.ssh/id_ed25519_oyjthub`（GitHub 标题 `lc-server-embodiedflow`）
- SSH 别名: `git@github-oyjt-hub:oyjt-hub/embodiedflow.git`
- 推送: `cd ~/workspace/embodiedflow && git push origin main`

无需 token，也无需交互输入。
