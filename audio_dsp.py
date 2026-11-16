"""Pure numpy/scipy audio cleaning, scoring and mastering.

ZipVoice clones timbre + prosody from the reference clip.  A clipped, noisy
or badly levelled reference produces a flat, "robotic" clone even with the
best model, so every reference goes through :func:`clean_reference` and
every output through :func:`master_output`.

Conventions: mono float32 in [-1, 1], sample rate in Hz.  All functions are
defensive: empty/short input returns input unchanged (callers validate
duration separately).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def peak(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _resample_poly(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(from_sr), int(to_sr))
    return signal.resample_poly(x, to_sr // g, from_sr // g).astype(np.float32)


def resample_to(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    return _resample_poly(np.asarray(x, dtype=np.float32), from_sr, to_sr)


def apply_fade(x: np.ndarray, sr: int, fade_ms: float = 10.0) -> np.ndarray:
    """Short raised-cosine fade in/out to remove clicks at joins."""
    x = np.asarray(x, dtype=np.float32).copy()
    n = max(1, int(sr * fade_ms / 1000.0))
    if x.size < 2 * n:
        n = max(1, x.size // 4)
    if n <= 1 or x.size == 0:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


# ---------------------------------------------------------------------------
# Reference cleaning
# ---------------------------------------------------------------------------


def trim_silence(
    x: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    frame_ms: float = 20.0,
    pad_ms: float = 120.0,
) -> np.ndarray:
    """Trim leading/trailing silence (energy-based) keeping a small pad."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, x.size // frame)
    energies = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n_frames)],
        dtype=np.float64,
    )
    if energies.size == 0 or float(np.max(energies)) <= 0:
        return x
    threshold = float(np.max(energies)) * (10 ** (-top_db / 20.0))
    active = np.flatnonzero(energies > threshold)
    if active.size == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    begin = max(0, int(active[0]) * frame - pad)
    end = min(x.size, (int(active[-1]) + 1) * frame + pad)
    return x[begin:end].copy()


def highpass(x: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """Remove rumble / handling noise below speech fundamentals."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < 64 or cutoff <= 0 or cutoff >= sr / 2:
        return x.copy()
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def soft_declip(x: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Repair clipped peaks with a gentle tanh soft-clip.

    Hard-clipped references (peak >= 1.0, e.g. phone recordings) teach the
    model a harsh timbre.  We first scale down, then soft-clip the residue.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    if x.size == 0:
        return x
    p = peak(x)
    if p <= 1.0:
        return x
    x = x * (0.98 / p)
    # tanh drive only affects samples near the ceiling.
    drive = 1.6
    x = np.tanh(x * drive) / np.tanh(drive * ceiling) * ceiling
    return x.astype(np.float32)


def spectral_gate_denoise(
    x: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop: int = 256,
    strength: float = 0.6,
) -> np.ndarray:
    """Conservative STFT noise gate for room tone / hiss.

    Estimates the noise floor from the quietest 20% of frames and attenuates
    frames whose energy is close to that floor.  ``strength`` in [0, 1];
    0.6 is deliberately light so speech onsets are never gated.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < n_fft * 2 or strength <= 0:
        return x.copy()
    try:
        _, _, spec = signal.stft(
            x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
            boundary="zeros", padded=True,
        )
    except Exception:
        return x.copy()
    mag = np.abs(spec)
    if mag.shape[1] < 4:
        return x.copy()
    frame_energy = np.mean(mag**2, axis=0)
    floor = float(np.percentile(frame_energy, 20))
    if floor <= 0:
        return x.copy()
    # Gain per frame: 1 well above floor, down to (1-strength) at floor.
    ratio = frame_energy / max(floor, 1e-12)
    gain = np.clip(1.0 - strength * np.exp(-ratio / 4.0), 1.0 - strength, 1.0)
    # Smooth gains to avoid musical noise.
    gain = np.convolve(gain, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    spec = spec * gain[np.newaxis, :]
    _, restored = signal.istft(
        spec, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        boundary=True,
    )
    restored = np.asarray(restored, dtype=np.float32)
    if restored.size != x.size:
        restored = restored[: x.size] if restored.size > x.size else np.pad(
            restored, (0, x.size - restored.size)
        )
    return restored


def normalize_rms(x: np.ndarray, target_rms: float = 0.15, peak_limit: float = 0.95) -> np.ndarray:
    """Scale to a target RMS, then peak-limit (never boosts into clipping)."""
    x = np.asarray(x, dtype=np.float32).copy()
    if x.size == 0:
        return x
    r = rms(x)
    if r <= 1e-9:
        return x
    x = x * (target_rms / r)
    p = peak(x)
    if p > peak_limit:
        x = x * (peak_limit / p)
    return x.astype(np.float32)


def clean_reference(
    x: np.ndarray,
    sr: int,
    target_sr: int = 24000,
    target_rms: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """Full reference pipeline: resample -> de-clip -> HP -> denoise -> trim -> level.

    Returns (cleaned_samples, report).  The report records what was fixed so
    the UI can explain quality to the user.
    """
    report: dict = {"actions": []}
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    if x.size == 0:
        return x, report
    if sr != target_sr:
        x = resample_to(x, sr, target_sr)
        sr = target_sr
        report["actions"].append(f"重采样至 {target_sr} Hz")
    if peak(x) > 1.0:
        x = soft_declip(x)
        report["actions"].append("削波修复（峰值过高，已软限幅）")
    x = highpass(x, sr, cutoff=80.0)
    x = spectral_gate_denoise(x, sr, strength=0.55)
    trimmed = trim_silence(x, sr)
    if trimmed.size < x.size:
        report["actions"].append("去除首尾静音")
    x = trimmed
    x = normalize_rms(x, target_rms=target_rms, peak_limit=0.95)
    report["actions"].append(f"响度归一（RMS {target_rms:.2f}）")
    return x.astype(np.float32), report


# ---------------------------------------------------------------------------
# Reference scoring
# ---------------------------------------------------------------------------


def score_reference(x: np.ndarray, sr: int) -> dict:
    """Heuristic 0-100 quality score for a reference clip.

    ZipVoice needs: single speaker, clean, 5-15 s, moderate level, no
    clipping.  The score is advisory (shown in the UI) and never blocks use.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    duration = x.size / max(1, sr)
    p = peak(x)
    r = rms(x)
    warnings: list[str] = []
    tips: list[str] = []
    score = 100.0

    if duration < 3:
        score -= 40
        warnings.append("片段不足 3 秒，音色信息太少")
        tips.append("建议录 6–15 秒完整句子")
    elif duration < 5:
        score -= 12
        tips.append("5 秒以下可用，但 6–15 秒更稳")
    elif duration > 20:
        score -= 10
        tips.append("超过 20 秒建议截取最清晰的一段")

    clip_ratio = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    if p > 1.0 or clip_ratio > 0.001:
        score -= 25
        warnings.append("录音峰值偏高/削波，已做软限幅修复")
        tips.append("重录时离麦克风稍远或调低输入音量")
    if r < 0.012:
        score -= 20
        warnings.append("录音音量较低，底噪占比偏高")
        tips.append("靠近麦克风重录，或在安静房间录制")
    elif r < 0.03:
        score -= 8
        tips.append("音量偏低，克隆可能偏弱")

    # Silence ratio (20 ms frames below -45 dBFS).
    frame = max(1, int(sr * 0.02))
    n_frames = max(1, x.size // frame)
    energies = np.array(
        [np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n_frames)]
    )
    silence_ratio = float(np.mean(energies < 0.0056)) if energies.size else 0.0
    if silence_ratio > 0.45:
        score -= 15
        warnings.append("静音占比过高，参考信息密度低")
        tips.append("选择连续说话的片段，少留长停顿")
    elif silence_ratio > 0.3:
        score -= 6

    # Crude SNR: speech frames vs quietest-frame floor.
    if energies.size >= 4:
        floor = float(np.percentile(energies, 10))
        speech = float(np.percentile(energies, 90))
        snr_db = 20 * math.log10(max(speech, 1e-9) / max(floor, 1e-9))
        if snr_db < 12:
            score -= 15
            warnings.append("信噪比偏低，环境噪声明显")
            tips.append("关掉风扇/空调，在安静环境重录")
        elif snr_db < 18:
            score -= 6
    else:
        snr_db = 0.0

    score = int(max(0, min(100, round(score))))
    if score >= 80:
        verdict = "优秀"
    elif score >= 60:
        verdict = "可用"
    elif score >= 40:
        verdict = "一般"
    else:
        verdict = "较差"
    return {
        "score": score,
        "verdict": verdict,
        "duration": round(duration, 3),
        "peak": round(p, 4),
        "rms": round(r, 4),
        "clip_ratio": round(clip_ratio, 5),
        "silence_ratio": round(silence_ratio, 3),
        "snr_db": round(float(snr_db), 1),
        "warnings": warnings,
        "tips": tips,
    }


# ---------------------------------------------------------------------------
# Output mastering + joining
# ---------------------------------------------------------------------------


def _peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Single peaking biquad (RBJ cookbook)."""
    if gain_db == 0 or x.size < 64:
        return x
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b = np.array([1 + alpha * a, -2 * cos_w0, 1 - alpha * a], dtype=np.float64)
    aa = np.array([1 + alpha / a, -2 * cos_w0, 1 - alpha / a], dtype=np.float64)
    b, aa = b / aa[0], aa / aa[0]
    return signal.lfilter(b, aa, x).astype(np.float32)


def _lowshelf(x: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    if gain_db == 0 or x.size < 64:
        return x
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / 2 * math.sqrt((a + 1 / a) * (1 / 0.7071 - 1) + 2)
    cos_w0 = math.cos(w0)
    sqrt_a = math.sqrt(a)
    b = np.array(
        [a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha),
         -2 * a * ((a - 1) + (a + 1) * cos_w0),
         a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)],
        dtype=np.float64,
    )
    aa = np.array(
        [(a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha,
         2 * ((a - 1) - (a + 1) * cos_w0),
         (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha],
        dtype=np.float64,
    )
    b, aa = b / aa[0], aa / aa[0]
    return signal.lfilter(b, aa, x).astype(np.float32)


def _soft_compress(x: np.ndarray, threshold: float = 0.35, ratio: float = 2.5) -> np.ndarray:
    """Memoryless soft-knee compressor (gentle, for voice)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    ax = np.abs(x)
    # Smooth envelope (5 ms) so gain changes don't click.
    env = np.convolve(ax, np.ones(128, dtype=np.float64) / 128.0, mode="same")
    over = np.maximum(env - threshold, 0.0)
    gain = 1.0 - (1.0 - 1.0 / ratio) * (over / np.maximum(env, 1e-9))
    gain = np.clip(gain, 0.35, 1.0)
    return (x * gain).astype(np.float32)


def master_output(
    x: np.ndarray,
    sr: int,
    strength: float = 1.0,
    target_rms: float = 0.16,
) -> np.ndarray:
    """Gentle mastering chain: HP -> warmth shelf -> presence -> compress -> level.

    Raw Vocos output can sound thin/flat.  This chain adds a little body and
    forward presence without changing timbre identity.  ``strength`` 0
    returns input unchanged; 1 is the default subtle setting.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    if x.size == 0 or strength <= 0:
        return x
    strength = float(min(1.5, max(0.0, strength)))
    x = highpass(x, sr, cutoff=70.0)
    x = _lowshelf(x, sr, freq=220.0, gain_db=1.2 * strength)
    x = _peaking_eq(x, sr, freq=3200.0, gain_db=1.6 * strength, q=0.9)
    x = _peaking_eq(x, sr, freq=6500.0, gain_db=0.8 * strength, q=1.2)
    x = _soft_compress(x, threshold=0.35, ratio=2.0 + 0.5 * strength)
    x = normalize_rms(x, target_rms=target_rms, peak_limit=0.95)
    x = apply_fade(x, sr, fade_ms=8.0)
    return x.astype(np.float32)


def join_chunks(
    chunks: list[tuple[np.ndarray, float]],
    sr: int,
    crossfade_ms: float = 15.0,
) -> np.ndarray:
    """Join (samples, pause_after_ms) chunks with pauses + crossfades.

    Pauses are inserted as true silence (natural breath gap) rather than
    stretching the model's own trailing silence, which avoids the "even,
    metronomic" robotic rhythm of single-shot long synthesis.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    fade_n = max(1, int(sr * crossfade_ms / 1000.0))
    out = np.asarray(chunks[0][0], dtype=np.float32).copy()
    # pause_ms on each entry is the pause *after* that chunk.
    for index in range(1, len(chunks)):
        samples = np.asarray(chunks[index][0], dtype=np.float32)
        pause_ms = float(chunks[index - 1][1])
        pause_n = max(0, int(sr * max(0.0, pause_ms) / 1000.0))
        if pause_n:
            out = np.concatenate([out, np.zeros(pause_n, dtype=np.float32)])
        # Crossfade the boundary to hide any level discontinuity.
        overlap = min(fade_n, out.size, samples.size)
        if overlap > 1:
            ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            out[-overlap:] = out[-overlap:] * (1.0 - ramp) + samples[:overlap] * ramp
            out = np.concatenate([out, samples[overlap:]])
        else:
            out = np.concatenate([out, samples])
    return out.astype(np.float32)


__all__ = [
    "apply_fade",
    "clean_reference",
    "highpass",
    "join_chunks",
    "master_output",
    "normalize_rms",
    "peak",
    "resample_to",
    "rms",
    "score_reference",
    "soft_declip",
    "spectral_gate_denoise",
    "trim_silence",
]
