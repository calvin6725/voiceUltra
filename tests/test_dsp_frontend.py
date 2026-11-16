"""Unit tests for audio_dsp / text_frontend / hardware (no models needed).

Run:  python tests/test_dsp_frontend.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_dsp
import hardware
import text_frontend

SR = 24000


def _speech_like(seconds=6.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    # Formant-ish synthetic "speech": harmonic stack with slow AM.
    x = (
        0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.3 * np.sin(2 * np.pi * 240 * t + 0.3)
        + 0.2 * np.sin(2 * np.pi * 480 * t + 0.7)
    ) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = x * 0.35 + rng.normal(0, 0.005, size=x.shape)
    return x.astype(np.float32)


def test_clean_reference_repairs_clipping():
    x = _speech_like() * 4.5  # hard-clipped
    assert audio_dsp.peak(x) > 1.0
    cleaned, report = audio_dsp.clean_reference(x, SR)
    assert cleaned.size > 0
    assert audio_dsp.peak(cleaned) <= 0.96, audio_dsp.peak(cleaned)
    assert audio_dsp.rms(cleaned) > 0.05
    assert report["actions"]


def test_score_reference_flags_bad_clip():
    good = _speech_like()
    bad = _speech_like() * 4.5
    good_score = audio_dsp.score_reference(good, SR)
    bad_score = audio_dsp.score_reference(bad, SR)
    assert good_score["score"] > bad_score["score"]
    assert bad_score["score"] < 80
    assert good_score["duration"] > 5


def test_master_output_levels_and_fades():
    x = _speech_like(seconds=2.0) * 0.2
    y = audio_dsp.master_output(x, SR, strength=1.0)
    assert y.shape == x.shape
    assert audio_dsp.peak(y) <= 0.96
    assert audio_dsp.rms(y) > audio_dsp.rms(x)  # levelled up
    # Fades: first/last samples near zero.
    assert abs(float(y[0])) < 0.05
    assert abs(float(y[-1])) < 0.05


def test_join_chunks_inserts_pause():
    a = _speech_like(seconds=1.0)
    b = _speech_like(seconds=1.0, seed=1)
    joined = audio_dsp.join_chunks([(a, 200.0), (b, 0.0)], SR)
    expected = a.size + int(SR * 0.2) + b.size
    assert abs(joined.size - expected) <= 2 * int(SR * 0.015)


def test_normalize_and_chunk():
    text = "  今天我们换一种方式讲故事。声音的细节、说话的节奏，还有每句话之间的停顿，都值得认真对待！  "
    norm = text_frontend.normalize_text(text)
    assert norm.endswith(("！", "!"))
    chunks = text_frontend.chunk_text(norm)
    assert len(chunks) >= 1
    assert all(len(c["text"]) <= 140 for c in chunks)
    assert chunks[-1]["pause_ms"] == 0
    # Emphasis markers stripped, pause marker honoured.
    assert "*" not in text_frontend.normalize_text("**你好**")
    split = text_frontend.split_sentences("第一句。[pause]第二句？")
    assert len(split) == 2


def test_prosody_variation_bounded_and_deterministic():
    v1 = text_frontend.prosody_speed_variation(3, 1.0)
    v2 = text_frontend.prosody_speed_variation(3, 1.0)
    assert v1 == v2
    assert 0.75 <= v1 <= 1.35
    assert abs(v1 - 1.0) <= 0.021


def test_hardware_recommendations_sane():
    threads = hardware.recommended_threads()
    assert 2 <= threads <= 8
    provider = hardware.recommended_provider()
    assert provider in ("cpu", "directml", "openvino", "cuda", "coreml")
    chain = hardware.provider_fallback_chain()
    assert chain and chain[-1] == "cpu"
    hw = hardware.detect_hardware()
    assert "cpu" in hw and "recommended_threads" in hw


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
