from __future__ import annotations

from .common import *
from .config import *
from .signal import *



def calibration_settings(cfg: Config) -> dict:
    """模板的采样、通道、相位和预处理契约；不包含测试标签或目标顺序。"""
    return {
        "frequencies": BENCHMARK_FREQUENCIES_HZ.tolist(),
        "phases": [t.phase_rad for t in TARGETS],
        "channel_indices": list(cfg.channel_indices),
        "channel_positions": list(cfg.channel_positions),
        "target_fs": cfg.target_fs, "window_s": cfg.window_s,
        "response_delay_s": cfg.response_delay_s,
        "timestamp_mode": cfg.timestamp_mode,
        "timestamp_processing": ("mne_lsl_streamlsl_clocksync_dejitter_monotize_v1"
                                 if cfg.timestamp_mode == "lsl" else "legacy_sample_clock"),
        "device_timestamp_lag_s": cfg.device_timestamp_lag_s,
        "n_harmonics": cfg.n_harmonics, "notch_hz": cfg.notch_hz,
        "filter_bands": [list(b) for b in cfg.filter_bands],
        "regularization": cfg.cca_regularization,
        "line_regression_hz": cfg.line_regression_hz,
        "a": cfg.weight_a, "b": cfg.weight_b,
    }

def _calibration_rows(x: np.ndarray) -> np.ndarray:
    """标准化校准矩阵，并要求所有通道均有效。"""
    z, active = _normalised_rows(x)
    if len(active) != len(x):
        raise DegenerateSignalError("个人模板要求所有校准通道有效")
    return z

def _calibration_whitener(x: np.ndarray, reg: float) -> np.ndarray:
    """计算个人校准特征使用的协方差白化矩阵。"""
    cov = x @ x.T / (x.shape[-1] - 1)
    cov += reg * np.trace(cov) / len(cov) * np.eye(len(cov))
    return _inverse_sqrt_covariance(cov, reg)

def _projected_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按最后一个轴计算成对投影信号的相关系数。"""
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.clip(np.divide(np.sum(x * y, axis=-1), denominator,
                            out=np.zeros_like(denominator), where=denominator > 0), -1, 1)

class PersonalCalibration:
    """固定四特征 eCCA + 原滤波器组权重，所有40目标始终参与分类。

    参照 Nakanishi et al. 2015, doi:10.1371/journal.pone.0140703，式14–15。
    每个训练窗先逐通道标准化；各子带再次标准化后逐类平均。
    三种CCA空间滤波保留投影相关的正负号，sum(sign(r)*r**2)融合。
    不训练阈值、不自学习、不使用当前真实标签；不会把分数解释为概率。
    """
    schema_version = 1

    def __init__(self, templates: np.ndarray, counts: np.ndarray, metadata: dict):
        """验证并冻结个人模板，同时预计算评分所需的空间滤波量。"""
        self._metadata = json.loads(json.dumps(metadata, allow_nan=False))
        m = self._metadata
        if m.get("schema_version") != self.schema_version or m.get("algorithm") != "FB-eCCA-4":
            raise ValueError("不支持的个人模型版本")
        if not isinstance(m.get("participant_id"), str) or not m["participant_id"].strip():
            raise ValueError("个人模型缺少受试者编号")
        s = m["settings"]
        self._settings = s
        _validate_filter_bands(s["filter_bands"], s["target_fs"], s["target_fs"])
        _validate_harmonics(s["n_harmonics"], np.asarray(s["frequencies"]), s["target_fs"])
        _validate_line_regression_hz(s["line_regression_hz"], s["target_fs"])
        for value in (s["regularization"], s["a"], s["b"], s["response_delay_s"]):
            if not np.isfinite(value) or value < 0:
                raise ValueError("个人模型正则化、权重或时延参数异常")
        if not np.isfinite(s["device_timestamp_lag_s"]):
            raise ValueError("个人模型设备时延参数异常")
        if s["notch_hz"] is not None and not 0 < s["notch_hz"] < s["target_fs"] / 2:
            raise ValueError("个人模型陷波参数异常")
        expected = (len(s["filter_bands"]), len(TARGETS), len(s["channel_positions"]),
                    sample_count(s["window_s"], s["target_fs"]))
        templates = np.array(templates, dtype=float, copy=True)
        counts = np.asarray(counts)
        if (templates.shape != expected or not np.isfinite(templates).all()
                or counts.shape != (len(TARGETS),) or counts.dtype.kind not in "iu"
                or np.any(counts < 1) or expected[-1] < 32
                or m.get("training_trials") != int(counts.sum())
                or len(s["channel_indices"]) != expected[2]
                or s["frequencies"] != BENCHMARK_FREQUENCIES_HZ.tolist()
                or s["phases"] != [t.phase_rad for t in TARGETS]):
            raise ValueError("个人模型模板、类别覆盖或目标表异常")
        _normalise_channel_indices(s["channel_indices"])
        for band in templates:
            for template in band:
                _calibration_rows(template)
        if (not np.allclose(templates.mean(-1), 0, atol=1e-8)
                or not np.allclose(templates.std(-1), 1, atol=1e-8)):
            raise ValueError("个人模板须逐通道中心化/标准化")
        self.templates = templates
        self.counts = np.array(counts, dtype=np.int64, copy=True)
        reg = s["regularization"]
        refs = np.array([_calibration_rows(r) for r in generate_reference_signals(
            s["frequencies"], s["target_fs"], expected[-1], s["n_harmonics"])])
        self._white_refs = np.array([_calibration_whitener(r, reg) @ r for r in refs])
        self._wt = np.array([[_calibration_whitener(t, reg) for t in band] for band in templates])
        self._white_templates = self._wt @ templates
        u, _, _ = np.linalg.svd(self._white_templates @ self._white_refs.swapaxes(-1, -2)
                                / (expected[-1] - 1), full_matrices=False)
        self._template_filters = np.einsum("bkij,bkj->bki", self._wt, u[:, :, :, 0])
        self._projected_templates = np.einsum("bkc,bkct->bkt", self._template_filters, templates)
        for value in (self.templates, self.counts, self._white_refs, self._wt,
                      self._white_templates, self._template_filters, self._projected_templates):
            value.setflags(write=False)
        digest = hashlib.sha256(json.dumps(m, sort_keys=True, allow_nan=False).encode())
        digest.update(self.templates.astype("<f8").tobytes())
        digest.update(self.counts.astype("<i8").tobytes())
        self.model_id = digest.hexdigest()

    @property
    def metadata(self) -> dict:
        """返回模型元数据的深拷贝，避免外部修改内部状态。"""
        return json.loads(json.dumps(self._metadata))

    @classmethod
    def fit(cls, windows: Sequence[np.ndarray], labels: Sequence[int],
            fs: float | Sequence[float],
            cfg: Config, *, sources: Sequence[dict] = ()) -> PersonalCalibration:
        """从覆盖全部目标的已对齐校准窗训练个人 eCCA 模板。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持个人模板训练；短分析窗不能代替完整历史预处理")
        cfg.validate()
        if not cfg.participant_id or not cfg.participant_id.strip():
            raise ValueError("训练个人模型必须声明受试者编号")
        y = np.asarray(labels)
        if (y.ndim != 1 or y.dtype.kind not in "iu" or len(y) != len(windows)
                or np.any((y < 1) | (y > len(TARGETS)))):
            raise ValueError("校准标签必须为与窗口一一对应的1–40整数")
        counts = np.bincount(y, minlength=len(TARGETS) + 1)[1:]
        if np.any(counts == 0):
            raise ValueError(f"个人模板缺少目标：{(np.flatnonzero(counts == 0) + 1).tolist()}")
        try:
            sampling_rates = np.asarray(fs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("校准窗采样率须为数值") from exc
        if sampling_rates.ndim == 0:
            sampling_rates = np.full(len(windows), float(sampling_rates))
        if (sampling_rates.shape != (len(windows),)
                or not np.isfinite(sampling_rates).all()
                or np.any(sampling_rates <= 0)):
            raise ValueError("校准窗采样率须与窗口一一对应且为有限正数")
        banks = []
        for x, window_fs in zip(windows, sampling_rates):
            x = _validate_eeg_ct(x)
            if x.shape != (len(cfg.channel_indices), sample_count(cfg.window_s, window_fs)):
                raise ValueError("校准窗的通道数/长度与配置不一致；须提供已对齐的完整窗")
            bands = _validate_filter_bands(cfg.filter_bands, window_fs, cfg.target_fs)
            _validate_harmonics(cfg.n_harmonics, BENCHMARK_FREQUENCIES_HZ, window_fs)
            _validate_line_regression_hz(cfg.line_regression_hz, min(window_fs, cfg.target_fs))
            if cfg.line_regression_hz is not None:
                x = _regress_line_noise(x, window_fs, cfg.line_regression_hz)
            _, bank = _preprocess_validated_window(_calibration_rows(x), window_fs, cfg.window_s,
                cfg.target_fs, cfg.notch_hz, bands)
            banks.append([_calibration_rows(b) for b in bank])
        banks = np.asarray(banks)
        templates = np.array([[_calibration_rows(b) for b in banks[y == k].mean(0)]
                              for k in range(1, len(TARGETS) + 1)]).swapaxes(0, 1)
        return cls(templates, counts, {
            "schema_version": cls.schema_version, "algorithm": "FB-eCCA-4",
            "participant_id": cfg.participant_id, "settings": calibration_settings(cfg),
            "sources": list(sources), "training_trials": len(y),
        })

    def validate_config(self, cfg: Config) -> None:
        """确认运行配置与模型的受试者及预处理契约一致。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持个人模型；旧模板仅适用于 epoch 预处理")
        if cfg.participant_id != self._metadata["participant_id"]:
            raise ValueError("个人模型受试者与 --participant 不一致")
        if (cfg.timestamp_mode == "lsl" and
                self._settings.get("timestamp_processing") != "mne_lsl_streamlsl_clocksync_dejitter_monotize_v1"):
            raise ValueError("个人模型使用旧接收路径；请用MNE-LSL重新完成40目标校准")
        if calibration_settings(cfg) != self._settings:
            raise ValueError("个人模型的通道、采样、时延、相位或滤波配置不匹配")
        if cfg.rejection_enabled and cfg.rejection_calibration_id != self.model_id:
            raise ValueError("个人模型改变了分数尺度；拒识阈值必须重新标定并绑定此模型ID")

    def validate_decoder(self, frequencies, fs, window_s, harmonics, notch, bands,
                         reg, line, a, b, channels) -> None:
        """确认一次解码调用的参数与模型训练设置完全一致。"""
        s = self._settings
        actual = [list(frequencies), fs, window_s, harmonics, notch, [list(v) for v in bands],
                  reg, line, a, b, channels]
        expected = [s["frequencies"], s["target_fs"], s["window_s"], s["n_harmonics"],
                    s["notch_hz"], s["filter_bands"], s["regularization"],
                    s["line_regression_hz"], s["a"], s["b"], len(s["channel_positions"])]
        if actual != expected:
            raise ValueError("分类参数与个人模型不匹配，拒绝套用模板")

    def score(self, bank: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算滤波器组 eCCA 融合分数及四类中间特征。"""
        bank, weights = np.asarray(bank, dtype=float), np.asarray(weights, dtype=float)
        expected = (self.templates.shape[0], *self.templates.shape[2:])
        if (bank.shape != expected or not np.isfinite(bank).all()
                or weights.shape != (expected[0],) or not np.isfinite(weights).all()
                or np.any(weights < 0)):
            raise ValueError("个人模板评分的子带形状或权重异常")
        features = []
        reg = self._settings["regularization"]
        for k, band in enumerate(bank):
            x = _calibration_rows(band)
            wx = _calibration_whitener(x, reg)
            xx = wx @ x
            u, singular, _ = np.linalg.svd(xx @ self._white_refs.swapaxes(-1, -2)
                                         / (x.shape[-1] - 1), full_matrices=False)
            ax = u[:, :, 0] @ wx
            ut, _, _ = np.linalg.svd(xx @ self._white_templates[k].swapaxes(-1, -2)
                                    / (x.shape[-1] - 1), full_matrices=False)
            at = ut[:, :, 0] @ wx
            template = self.templates[k]
            features.append(np.array([
                np.clip(singular[:, 0], 0, 1),
                _projected_correlation(at @ x, np.einsum("kc,kct->kt", at, template)),
                _projected_correlation(ax @ x, np.einsum("kc,kct->kt", ax, template)),
                _projected_correlation(self._template_filters[k] @ x, self._projected_templates[k]),
            ]))
        features = np.asarray(features)
        return weights @ np.sum(np.sign(features) * features ** 2, axis=1), features

    def save(self, path: str | Path) -> None:
        """以原子 NPZ 文件保存模板、计数、元数据和完整性标识。"""
        # 延迟导入避免 recording -> calibration 的模块循环依赖。
        from .recording import SessionRecorder

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        SessionRecorder._atomic_npz(path, {
            "templates": self.templates, "counts": self.counts,
            "metadata_json": np.array(json.dumps(self._metadata, ensure_ascii=False, allow_nan=False)),
            "model_id": np.array(self.model_id),
        })

    @classmethod
    def load(cls, path: str | Path, cfg: Config) -> PersonalCalibration:
        """从 NPZ、会话模型 PKL 或会话 ZIP 加载并验证个人模型。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持加载个人模型；请使用无模板 FBCCA")
        path = Path(path)
        if path.suffix.lower() in (".pkl", ".zip"):
            class DataOnlyUnpickler(pickle.Unpickler):
                """禁止通过 pickle 反序列化任意 Python 类。"""

                def find_class(self, module: str, name: str) -> Any:
                    """拒绝所有需要导入类或全局对象的 pickle 条目。"""
                    raise pickle.UnpicklingError(f"不允许加载PKL对象：{module}.{name}")

            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    # 精简会话ZIP使用 model.pkl；旧版完整ZIP使用 <会话名>_model.pkl。
                    member = "model.pkl" if "model.pkl" in archive.namelist() else path.stem + "_model.pkl"
                    with archive.open(member) as handle:
                        bundle = DataOnlyUnpickler(handle).load()
            else:
                with path.open("rb") as handle:
                    bundle = DataOnlyUnpickler(handle).load()
            if not isinstance(bundle, dict) or bundle.get("format") != "fbcca_keyboard_model_v1":
                raise ValueError("不支持的会话PKL模型格式")
            if bundle.get("training_status") != "trained":
                raise ValueError("该会话PKL没有完整的40类个人模型：" +
                                 str(bundle.get("training_reason", "样本不足")))
            calibration = bundle.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError("会话PKL缺少个人模板")
            shape = calibration.get("templates_shape")
            raw = calibration.get("templates_float64_le")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 4
                    or not all(isinstance(v, int) and v > 0 for v in shape)
                    or not isinstance(raw, bytes)
                    or len(raw) != math.prod(shape) * 8):
                raise ValueError("会话PKL模板形状或字节长度异常")
            templates = np.frombuffer(raw, dtype="<f8").reshape(shape)
            model = cls(templates, np.asarray(calibration["counts"]), calibration["metadata"])
            if model.model_id != calibration.get("model_id"):
                raise ValueError("个人模型完整性校验失败")
        else:
            with np.load(path, allow_pickle=False) as saved:
                model = cls(saved["templates"], saved["counts"], json.loads(saved["metadata_json"].item()))
                if model.model_id != saved["model_id"].item():
                    raise ValueError("个人模型完整性校验失败")
        model.validate_config(cfg)
        return model



__all__ = [name for name in globals() if not name.startswith('__')]

