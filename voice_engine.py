"""Local voice cloning: SenseVoice ASR + reference-conditioned ZipVoice TTS.

Hardware target: Intel Core Ultra 5 125H + 32 GB RAM + Intel Arc iGPU.
Inference runs on CPU by default and uses the Arc iGPU via DirectML /
OpenVINO when those ONNX Runtime providers are installed (see hardware.py).
A bad reference is the #1 cause of "robotic" clones, so references are
cleaned + scored (audio_dsp) and text is chunked by sentence with natural
pauses (text_frontend) before synthesis.
"""

from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import uuid

import imageio_ffmpeg
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:  # pragma: no cover - exercised only without deps
    sherpa_onnx = None  # type: ignore

import audio_dsp
import hardware
import text_frontend

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "voice-data"
THREADS = hardware.recommended_threads()
PROVIDER = hardware.recommended_provider()
MODEL_NAMES = {
    "fast": "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
    "standard": "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia",
    "full": "sherpa-onnx-zipvoice-zh-en-emilia",
}
# Naturalness-first presets.  "fast" (INT8 + 4 steps) is kept only for
# quick previews; the default everywhere is "standard" or "full".
QUALITY_PRESETS = {
    "fast": {"label": "快速试听", "steps": 4, "guidance_scale": 1.0, "silence_scale": 1.0},
    "standard": {"label": "均衡", "steps": 8, "guidance_scale": 1.0, "silence_scale": 1.0},
    "full": {"label": "自然优先", "steps": 16, "guidance_scale": 2.0, "silence_scale": 1.0},
}
DEFAULT_QUALITY = "standard"
SAMPLE_RATE = 24000


class EngineBusy(RuntimeError):
    pass


class VoiceEngine:
    def __init__(self):
        DATA.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.model = None
        self.model_quality = None
        self.model_provider = None
        self.asr = None
        self.activity = {"stage": "idle", "progress": 0}

    @contextmanager
    def operation(self, stage):
        if not self.lock.acquire(blocking=False):
            raise EngineBusy("正在处理上一个任务，请稍后重试")
        self.activity = {"stage": stage, "progress": 0, "started": time.time()}
        try:
            yield
        finally:
            self.activity = {"stage": "idle", "progress": 0}
            self.lock.release()

    def status(self):
        hw = hardware.detect_hardware()
        return {
            "engine": "zipvoice",
            "provider": self.model_provider or PROVIDER,
            "requested_provider": PROVIDER,
            "threads": THREADS,
            "hardware": hw,
            "qualities": [
                name for name, folder in MODEL_NAMES.items()
                if (MODELS / folder / ".ready").is_file()
            ],
            "presets": QUALITY_PRESETS,
            "default_quality": DEFAULT_QUALITY,
            "recognition": self.asr_directory() is not None,
            "activity": self.activity,
            "profiles": self.profiles(),
        }

    def asr_directory(self):
        return next(
            (p for p in sorted(MODELS.glob("sherpa-onnx-sense-voice-*"), reverse=True)
             if p.is_dir() and (p / "tokens.txt").is_file()), None,
        )

    def _path(self, reference_id):
        try:
            if str(uuid.UUID(str(reference_id))) != reference_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ValueError("声音参考编号无效")
        return DATA / reference_id

    def reference(self, reference_id):
        path = self._path(reference_id)
        if not path.with_suffix(".json").is_file():
            raise ValueError("找不到声音参考，请重新导入录音")
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    def _save(self, item):
        path = self._path(item["id"]).with_suffix(".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def profiles(self):
        result = []
        for path in DATA.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("saved"):
                    result.append(item)
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda item: item.get("updatedAt", 0), reverse=True)

    def _decode(self, data, start, duration):
        """Decode + clean a reference clip. Returns (samples, report, peak, rms)."""
        handle, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(handle, "wb") as target:
                target.write(data)
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin",
                "-ss", str(start), "-i", path, "-t", str(duration),
                "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
            ]
            result = subprocess.run(
                command, capture_output=True, timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode:
                raise ValueError("无法读取音频，请选择有效的 MP3、WAV 或 M4A 文件")
            raw = np.frombuffer(result.stdout, dtype="<f4").copy()
        finally:
            Path(path).unlink(missing_ok=True)
        if len(raw) < 3 * SAMPLE_RATE:
            raise ValueError("参考片段至少需要 3 秒，建议使用 6 至 15 秒清晰人声")
        raw_peak = float(np.max(np.abs(raw))) if raw.size else 0.0
        raw_rms = float(np.sqrt(np.mean(raw**2))) if raw.size else 0.0
        if raw_rms < 0.001:
            raise ValueError("录音音量太低，请换一段清晰人声")
        # Score the raw clip first (so warnings reflect the original), then
        # clean it for cloning.  Clipped/noisy references are repaired, not
        # passed through verbatim.
        quality = audio_dsp.score_reference(raw, SAMPLE_RATE)
        samples, clean_report = audio_dsp.clean_reference(raw, SAMPLE_RATE)
        if len(samples) < 3 * SAMPLE_RATE:
            raise ValueError("去除首尾静音后人声不足 3 秒，请重新选择片段")
        warnings = list(quality.get("warnings", []))
        warnings.extend(clean_report.get("actions", []))
        return samples, {
            "quality": quality,
            "clean_actions": clean_report.get("actions", []),
            "warnings": warnings,
        }, raw_peak, raw_rms

    def transcribe(self, samples):
        directory = self.asr_directory()
        if directory is None or sherpa_onnx is None:
            return ""
        if self.asr is None:
            self.activity["stage"] = "loading_recognizer"
            model = next(directory.glob("model*.onnx"))
            self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(directory / "tokens.txt"),
                num_threads=THREADS, language="auto", use_itn=True,
            )
        self.activity["stage"] = "transcribing"
        stream = self.asr.create_stream()
        stream.accept_waveform(16000, resample_poly(samples, 2, 3).astype(np.float32))
        self.asr.decode_stream(stream)
        return re.sub(r"<\|[^|]*\|>", "", stream.result.text).strip()

    def import_reference(self, data, name, start=0, duration=12, transcript=None):
        if not math.isfinite(start) or not math.isfinite(duration):
            raise ValueError("参考片段时间无效")
        if start < 0 or not 3 <= duration <= 30:
            raise ValueError("请选择 3 至 30 秒的参考片段（建议 6 至 15 秒）")
        with self.operation("preparing_reference"):
            samples, report, peak, rms = self._decode(data, start, duration)
            text = self.transcribe(samples) if transcript is None else transcript.strip()
            reference_id = str(uuid.uuid4())
            sf.write(str(self._path(reference_id).with_suffix(".wav")),
                     samples, SAMPLE_RATE, subtype="PCM_16")
            item = {
                "id": reference_id, "name": str(name)[:100],
                "text": text, "duration": round(len(samples) / SAMPLE_RATE, 3),
                "sampleRate": SAMPLE_RATE, "warnings": report["warnings"],
                "quality": report["quality"],
                "clean_actions": report["clean_actions"],
                "peak": peak, "rms": rms, "saved": False,
                "updatedAt": time.time(),
            }
            self._save(item)
            return item

    def save_profile(self, reference_id, name, text):
        item = self.reference(reference_id)
        name, text = str(name).strip(), str(text).strip()
        if not name or len(name) > 50:
            raise ValueError("音色名称应为 1 至 50 个字")
        if not text or len(text) > 1000:
            raise ValueError("请填写参考片段实际说出的原话（最多 1000 字）")
        item.update(name=name, text=text, saved=True, updatedAt=time.time())
        self._save(item)
        return item

    def delete_profile(self, reference_id):
        with self.operation("deleting_profile"):
            self.reference(reference_id)
            self._path(reference_id).with_suffix(".json").unlink()
            self._path(reference_id).with_suffix(".wav").unlink(missing_ok=True)

    def _build_tts(self, quality, provider):
        directory = MODELS / MODEL_NAMES[quality]
        if quality == "full":
            encoder = directory / "text_encoder.onnx"
            decoder = directory / "fm_decoder.onnx"
        else:
            encoder = next(directory.glob("encoder*.onnx"))
            decoder = next(directory.glob("decoder*.onnx"))
        lexicon = directory / "lexicon.txt"
        if not lexicon.is_file():
            lexicon = MODELS / MODEL_NAMES["fast"] / "lexicon.txt"
        preset = QUALITY_PRESETS[quality]
        zipvoice_kwargs = {
            "tokens": str(directory / "tokens.txt"),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "lexicon": f"{lexicon},{MODELS / 'lexicon-en-us.txt'}",
            "vocoder": str(MODELS / "vocos_24khz.onnx"),
            "guidance_scale": preset["guidance_scale"],
        }
        # Optional ZipVoice knobs (present in newer sherpa-onnx).  Set only
        # when the binding supports them so older wheels keep working.
        try:
            probe = sherpa_onnx.OfflineTtsZipvoiceModelConfig(**zipvoice_kwargs)
            for attr, value in (("feat_scale", 1.0), ("t_shift", 0.1), ("target_rms", 0.1)):
                if hasattr(probe, attr):
                    setattr(probe, attr, value)
            zipvoice_cfg = probe
        except TypeError:
            zipvoice_cfg = sherpa_onnx.OfflineTtsZipvoiceModelConfig(**zipvoice_kwargs)
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                zipvoice=zipvoice_cfg,
                num_threads=THREADS, provider=provider, debug=False,
            )
        )
        if not config.validate():
            raise RuntimeError("克隆模型文件不完整，请重新下载")
        return sherpa_onnx.OfflineTts(config)

    def _load_model(self, quality):
        if quality not in MODEL_NAMES:
            raise ValueError("生成模式无效")
        directory = MODELS / MODEL_NAMES[quality]
        if not (directory / ".ready").is_file():
            raise ValueError("所选模型尚未安装，请先完成模型下载")
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx 未安装，请先安装运行环境")
        if self.model is not None and self.model_quality == quality:
            return self.model
        self.activity["stage"] = "loading_model"
        last_error = None
        for provider in hardware.provider_fallback_chain():
            try:
                self.model = self._build_tts(quality, provider)
                self.model_quality = quality
                self.model_provider = provider
                return self.model
            except Exception as exc:  # provider unavailable -> try next
                last_error = exc
                self.model = None
                continue
        raise RuntimeError(f"克隆模型加载失败（已尝试 CPU/GPU 均不可用）: {last_error}")

    def _synthesize_chunk(self, model, chunk_text, ref_samples, ref_sr, ref_text,
                          speed, steps, silence_scale):
        config = sherpa_onnx.GenerationConfig()
        config.reference_audio = ref_samples
        config.reference_sample_rate = ref_sr
        config.reference_text = ref_text
        config.speed = speed
        config.num_steps = steps
        config.silence_scale = silence_scale
        config.extra["min_words_in_sentence"] = "12"
        config.extra["max_words_in_sentence"] = "40"

        def progress(chunk, fraction):
            return 1

        generated = model.generate(chunk_text, config, progress)
        if not len(generated.samples):
            raise RuntimeError("克隆模型没有生成音频，请检查参考录音和原话")
        samples = np.asarray(generated.samples, dtype=np.float32)
        if not np.all(np.isfinite(samples)):
            raise RuntimeError("模型输出无效，请换一段清晰的参考录音重试")
        return samples, generated.sample_rate

    def generate(self, reference_id, text, quality=DEFAULT_QUALITY, speed=1.0,
                 steps=None, prosody="natural", pause_scale=1.0, mastering=True,
                 guidance_scale=None):
        """Synthesize chunked, mastered speech. Returns (wav_bytes, info).

        ``steps=None`` selects the preset default for ``quality`` (full=16,
        standard=8, fast=4).  ``prosody`` is currently informational
        (natural/broadcast/story) and reserved for future emotion control;
        chunking + pauses + mastering already deliver most of the gain.
        """
        text = text_frontend.normalize_text(str(text))
        if not text or len(text) > 3000:
            raise ValueError("配音文字应为 1 至 3000 个字")
        if not math.isfinite(speed) or not 0.75 <= speed <= 1.35:
            raise ValueError("语速应为 0.75 至 1.35 倍")
        if quality not in MODEL_NAMES:
            raise ValueError("生成模式无效")
        preset = QUALITY_PRESETS[quality]
        if steps is None:
            steps = preset["steps"]
        if steps not in (4, 8, 16):
            raise ValueError("采样步数无效")
        if pause_scale is None or not math.isfinite(pause_scale) or not 0.5 <= pause_scale <= 2.0:
            raise ValueError("停顿强度应为 0.5 至 2.0 倍")
        reference = self.reference(reference_id)
        if not reference["text"].strip():
            raise ValueError("请先核对并保存参考录音的原话")
        chunks = text_frontend.chunk_text(text, pause_scale=float(pause_scale))
        if not chunks:
            raise ValueError("配音文字为空")
        with self.operation("loading_model"):
            started = time.perf_counter()
            model = self._load_model(quality)
            ref_samples, ref_sr = sf.read(
                str(self._path(reference_id).with_suffix(".wav")), dtype="float32"
            )
            if ref_samples.ndim > 1:
                ref_samples = ref_samples[:, 0].astype(np.float32)
            # Back-compat: profiles saved before the cleaning pipeline may
            # still be clipped/uneven.  Repair them on the fly.
            if audio_dsp.peak(ref_samples) > 0.99:
                ref_samples = audio_dsp.soft_declip(ref_samples)
                ref_samples = audio_dsp.normalize_rms(ref_samples, target_rms=0.15)
            silence_scale = preset["silence_scale"]
            self.activity["stage"] = "synthesizing"
            pieces: list[tuple[np.ndarray, float]] = []
            out_sr = SAMPLE_RATE
            for index, chunk in enumerate(chunks):
                self.activity["progress"] = index / max(1, len(chunks))
                chunk_speed = text_frontend.prosody_speed_variation(index, float(speed))
                samples, out_sr = self._synthesize_chunk(
                    model, chunk["text"], ref_samples, ref_sr,
                    reference["text"], chunk_speed, steps, silence_scale,
                )
                pieces.append((samples, float(chunk["pause_ms"])))
            self.activity["progress"] = 1.0
            joined = audio_dsp.join_chunks(pieces, out_sr)
            raw_peak = float(np.max(np.abs(joined))) if joined.size else 0.0
            if mastering:
                joined = audio_dsp.master_output(joined, out_sr, strength=1.0)
            output_peak = float(np.max(np.abs(joined))) if joined.size else 0.0
            if output_peak > 0.98:
                joined = joined * (0.98 / output_peak)
            wave = io.BytesIO()
            sf.write(wave, joined, out_sr, format="WAV", subtype="PCM_16")
            elapsed = time.perf_counter() - started
            duration = len(joined) / out_sr if out_sr else 0.0
            return wave.getvalue(), {
                "elapsed": round(elapsed, 2),
                "duration": round(duration, 2),
                "rtf": round(elapsed / duration, 3) if duration else None,
                "quality": quality, "steps": steps, "threads": THREADS,
                "provider": self.model_provider or PROVIDER,
                "speed": speed, "prosody": prosody,
                "pause_scale": pause_scale, "mastering": bool(mastering),
                "chunks": len(chunks),
                "rawPeak": raw_peak,
            }
