
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

## ChatGPT API（可选）

安装 OpenAI Python SDK，并在当前终端设置 API 密钥：

```bash
.venv/bin/python -m pip install -e '.[llm]'
export OPENAI_API_KEY='你的API密钥'
export OPENAI_MODEL='gpt-5-mini'
```

应用层使用 `ssvep_fbcca.llm.OpenAIChatBackend` 发送已经确认的文本草稿。
API 调用不参与 EEG 解码；空闲扫描、拒识和发送确认仍由键盘输入状态机负责。

启动前请让 OpenBCI GUI 持续发布名为 `obci_eeg1`、类型为 `EEG` 的 LSL 流。只检查依赖和保存目录时执行：

```bash
./start_keyboard_dual_mode.command --diagnose --config config/default.toml
```

离线分析不连接 EEG 或显示窗口：

```bash
./start_keyboard_dual_mode.command --offline-session 会话.zip --offline-output offline_results
```
