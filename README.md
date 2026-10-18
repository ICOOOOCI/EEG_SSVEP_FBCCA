# EEG SSVEP FBCCA Keyboard

项目已采用 `src` 布局，实时应用按职责拆分为以下模块：

- `config.py`：实验、采集、解码、显示和持久化配置，以及 TOML 加载。
- `signal.py`：参考信号、质量检查、预处理、CCA/FBCCA 解码。
- `calibration.py`：个人 eCCA 模板的训练、校验和持久化。
- `acquisition.py`：MNE-LSL 连续采集、事件窗和时间轴检查。
- `events.py`：LSL Marker、试次记录和文本输入状态。
- `recording.py`：连续 EEG 分块、逐试次文件和会话 ZIP 导出。
- `quality.py` / `analysis.py`：频谱质控、统计摘要和传输预检。
- `ui.py`：PsychoPy 键盘布局与交互流程。
- `app.py`：应用编排、命令行参数和故障出口。

## 启动

```bash
./start_keyboard_dual_mode.command
```

也可以直接指定标准 TOML 配置：

```bash
./start_keyboard_dual_mode.command --config config/default.toml
```

新代码应从 `ssvep_fbcca.config`、`ssvep_fbcca.signal` 等职责模块导入，
不要再依赖已移除的旧单文件入口。

## 安装

```bash
python3.10 -m pip install -e '.[realtime]'
```

没有 EEG/显示设备时，可以使用 `--diagnose` 检查运行环境和保存目录；该检查
不会宣称 LSL 流或物理显示延迟已经验证。

