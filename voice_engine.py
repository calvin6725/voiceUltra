"""Local CPU speech recognition and reference-conditioned ZipVoice synthesis."""

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
import sherpa_onnx
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "voice-data"
THREADS = min(8, max(2, (os.cpu_count() or 4) // 2))
MODEL_NAMES = {
    "fast": "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
    "standard": "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia",
    "full": "sherpa-onnx-zipvoice-zh-en-emilia",
}


class EngineBusy(RuntimeError):
    pass


class VoiceEngine:
    def __init__(self):
        DATA.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.model = None
        self.model_quality = None
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
        return {
            "engine": "zipvoice", "provider": "cpu", "threads": THREADS,
            "qualities": [
                name for name, folder in MODEL_NAMES.items()
                if (MODELS / folder / ".ready").is_file()
            ],
            "recognition": self.asr_directory() is not None,
            "activity": self.activity, "profiles": self.profiles(),
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
        handle, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(handle, "wb") as target:
                target.write(data)
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin",
                "-ss", str(start), "-i", path, "-t", str(duration),
                "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1",
            ]
            result = subprocess.run(
                command, capture_output=True, timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode:
                raise ValueError("无法读取音频，请选择有效的 MP3、WAV 或 M4A 文件")
            samples = np.frombuffer(result.stdout, dtype="<f4").copy()
        finally:
            Path(path).unlink(missing_ok=True)
        if len(samples) < 3 * 24000:
            raise ValueError("参考片段至少需要 3 秒，建议使用 6 至 15 秒清晰人声")
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples**2)))
        if rms < 0.001:
            raise ValueError("录音音量太低，请换一段清晰人声")
        warnings = []
        if peak > 0.99:
            warnings.append("录音峰值偏高，已降低音量以避免转存时失真")
        if rms < 0.012:
            warnings.append("录音音量较低，请留意底噪")
        frame = 480
        energies = np.array([
            np.sqrt(np.mean(samples[i:i + frame] ** 2))
            for i in range(0, len(samples), frame)
        ])
        active = np.flatnonzero(energies > max(0.003, rms * 0.08))
        if len(active):
            begin = max(0, (int(active[0]) - 5) * frame)
            end = min(len(samples), (int(active[-1]) + 6) * frame)
            samples = samples[begin:end]
        if len(samples) < 3 * 24000:
            raise ValueError("去除首尾静音后人声不足 3 秒，请重新选择片段")
        if peak > 0.95:
            samples *= 0.95 / peak
        return samples, warnings, peak, rms

    def transcribe(self, samples):
        directory = self.asr_directory()
        if directory is None:
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
        if start < 0 or not 3 <= duration <= 20:
            raise ValueError("请选择 3 至 20 秒的参考片段")
        with self.operation("preparing_reference"):
            samples, warnings, peak, rms = self._decode(data, start, duration)
            text = self.transcribe(samples) if transcript is None else transcript.strip()
            reference_id = str(uuid.uuid4())
            sf.write(str(self._path(reference_id).with_suffix(".wav")),
                     samples, 24000, subtype="PCM_16")
            item = {
                "id": reference_id, "name": str(name)[:100],
                "text": text, "duration": round(len(samples) / 24000, 3),
                "sampleRate": 24000, "warnings": warnings,
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

    def _load_model(self, quality):
        if quality not in MODEL_NAMES:
            raise ValueError("生成模式无效")
        directory = MODELS / MODEL_NAMES[quality]
        if not (directory / ".ready").is_file():
            raise ValueError("所选模型尚未安装，请先完成模型下载")
        if self.model is not None and self.model_quality == quality:
            return self.model
        self.activity["stage"] = "loading_model"
        if quality == "full":
            encoder = directory / "text_encoder.onnx"
            decoder = directory / "fm_decoder.onnx"
        else:
            encoder = next(directory.glob("encoder*.onnx"))
            decoder = next(directory.glob("decoder*.onnx"))
        lexicon = directory / "lexicon.txt"
        if not lexicon.is_file():
            lexicon = MODELS / MODEL_NAMES["fast"] / "lexicon.txt"
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                    tokens=str(directory / "tokens.txt"),
                    encoder=str(encoder), decoder=str(decoder),
                    lexicon=f"{lexicon},{MODELS / 'lexicon-en-us.txt'}",
                    vocoder=str(MODELS / "vocos_24khz.onnx"),
                    guidance_scale=2.0 if quality == "full" else 1.0,
                ),
                num_threads=THREADS, provider="cpu", debug=False,
            )
        )
        if not config.validate():
            raise RuntimeError("克隆模型文件不完整，请重新下载")
        self.model = None
        self.model = sherpa_onnx.OfflineTts(config)
        self.model_quality = quality
        return self.model

    def generate(self, reference_id, text, quality="fast", speed=1.0, steps=4):
        text = str(text).strip()
        if not text or len(text) > 3000:
            raise ValueError("配音文字应为 1 至 3000 个字")
        if not math.isfinite(speed) or not 0.75 <= speed <= 1.35:
            raise ValueError("语速应为 0.75 至 1.35 倍")
        if steps not in (4, 8, 16):
            raise ValueError("采样步数无效")
        reference = self.reference(reference_id)
        if not reference["text"].strip():
            raise ValueError("请先核对并保存参考录音的原话")
        with self.operation("loading_model"):
            started = time.perf_counter()
            model = self._load_model(quality)
            samples, sample_rate = sf.read(
                str(self._path(reference_id).with_suffix(".wav")), dtype="float32"
            )
            config = sherpa_onnx.GenerationConfig()
            config.reference_audio = samples
            config.reference_sample_rate = sample_rate
            config.reference_text = reference["text"]
            config.speed = speed
            config.num_steps = steps
            config.silence_scale = 1.0
            config.extra["min_words_in_sentence"] = "12"
            config.extra["max_words_in_sentence"] = "40"
            self.activity["stage"] = "synthesizing"

            def progress(chunk, fraction):
                self.activity["progress"] = float(fraction)
                return 1

            generated = model.generate(text, config, progress)
            if not len(generated.samples):
                raise RuntimeError("克隆模型没有生成音频，请检查参考录音和原话")
            output_samples = np.asarray(generated.samples, dtype=np.float32)
            if not np.all(np.isfinite(output_samples)):
                raise RuntimeError("模型输出无效，请换一段清晰的参考录音重试")
            output_peak = float(np.max(np.abs(output_samples)))
            if output_peak > 0.98:
                output_samples = output_samples * (0.98 / output_peak)
            wave = io.BytesIO()
            sf.write(wave, output_samples, generated.sample_rate,
                     format="WAV", subtype="PCM_16")
            elapsed = time.perf_counter() - started
            return wave.getvalue(), {
                "elapsed": round(elapsed, 2),
                "duration": round(len(generated.samples) / generated.sample_rate, 2),
                "quality": quality, "steps": steps, "threads": THREADS,
                "speed": speed, "rawPeak": output_peak,
            }
