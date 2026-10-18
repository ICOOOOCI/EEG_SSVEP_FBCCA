# EEG SSVEP FBCCA 键盘

## macOS 启动

项目使用 64 位 CPython 3.10。首次运行先在项目目录执行：

```bash
python3.10 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[realtime]'
```

然后双击 `start_keyboard_dual_mode.command`，或执行：

```bash
./start_keyboard_dual_mode.command
```

启动前请让 OpenBCI GUI 持续发布名为 `obci_eeg1`、类型为 `EEG` 的 LSL 流。只检查依赖和保存目录时执行：

```bash
./start_keyboard_dual_mode.command --diagnose --config config/default.toml
```

离线分析不连接 EEG 或显示窗口：

```bash
./start_keyboard_dual_mode.command --offline-session 会话.zip --offline-output offline_results
```
