# Remote Control Skills

## 目的
通过 GitHub 仓库文件下发命令，让服务器定时拉取并执行，执行结果写入日志并回推到同一分支。

## 检测频率
- 采用自适应检测：
	- 检测到 GitHub 更新后，下次间隔为 30 秒。
	- 如果本次没有更新，则间隔依次递增：30 秒 -> 1 分钟 -> 2 分钟 -> 3 分钟 -> 4 分钟 -> 5 分钟。
	- 达到 5 分钟后不再继续增加。
	- systemd timer 会每 30 秒唤醒一次执行器，由执行器根据本地状态决定是否真正执行检测。

## 退出登录后的运行方式
- 之前使用的是用户级 systemd timer，如果系统没有开启 lingering，那么用户退出登录后，`systemctl --user` 管理的定时器会停止。
- 解决方式有两种：
	1. 开启 lingering（需要管理员权限）：`sudo loginctl enable-linger huyihan`
	2. 改用系统级 timer（推荐），通过 [scripts/remote_control/install_system.sh](scripts/remote_control/install_system.sh) 安装。

## 仅使用用户级权限的离线常驻方案
- 如果没有管理员权限，也可以使用 `nohup`/`tmux` 启动一个用户级后台守护进程。
- 推荐脚本：
	- 启动：[scripts/remote_control/start_user_daemon.sh](scripts/remote_control/start_user_daemon.sh)
	- 停止：[scripts/remote_control/stop_user_daemon.sh](scripts/remote_control/stop_user_daemon.sh)
	- 守护进程：[scripts/remote_control/user_daemon.sh](scripts/remote_control/user_daemon.sh)
- 这种方式不需要 sudo，但前提是系统不会在注销时强制清理该用户进程。
- 守护进程会每 30 秒调用一次执行器，执行器本身会根据自适应状态决定是否真正拉取和执行。

## 关键文件
- 命令文件：[control/commands.txt](control/commands.txt)
- 执行日志：[logs/remote-control.log](logs/remote-control.log)
- 执行器：[scripts/remote_control/runner.sh](scripts/remote_control/runner.sh)

## 使用流程
1. 在 GitHub 上切换到目标分支（例如 remote-control-setup）。
2. 编辑并提交命令到 [control/commands.txt](control/commands.txt)（每行一条命令）。
3. 等待系统在下一次定时检测时拉取并执行。
4. 查看结果：打开 [logs/remote-control.log](logs/remote-control.log)。
5. 执行完成后，命令会被自动清空并回推到同一分支。

## 命令书写规则
- 每行一条命令。
- 空行和以 # 开头的行会被忽略。
- 命令在服务器上以 `bash -lc` 方式执行，支持常见 shell 语法。

## 注意事项
- 请只在同一分支提交命令和查看结果。
- 如果看到日志里出现 “Push failed.”，说明仓库权限或分支保护导致回推失败。
- 如需修改检测频率，请调整 [scripts/remote_control/install.sh](scripts/remote_control/install.sh) 里的 timer 配置以及 [scripts/remote_control/runner.sh](scripts/remote_control/runner.sh) 里的自适应策略。
