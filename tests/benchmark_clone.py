import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_engine import VoiceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", default="fast")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reference", default=r"C:\Users\12495\Desktop\素材.mp3")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "试听样本"
    output.mkdir(exist_ok=True)
    engine = VoiceEngine()
    profile = next((p for p in engine.profiles() if p["name"] == "素材原声"), None)
    if profile is None:
        started = time.perf_counter()
        source = Path(args.reference)
        reference = engine.import_reference(source.read_bytes(), source.name)
        print("REFERENCE", json.dumps(reference, ensure_ascii=False), flush=True)
        print("ASR_SECONDS", round(time.perf_counter() - started, 2), flush=True)
        profile = engine.save_profile(reference["id"], "素材原声", reference["text"])
    (output / "参考原声.wav").write_bytes(
        engine._path(profile["id"]).with_suffix(".wav").read_bytes()
    )
    text = "今天我们换一种方式讲故事。声音的细节、说话的节奏，还有每句话之间的停顿，都值得认真对待。"
    data, info = engine.generate(profile["id"], text, args.quality, args.speed, args.steps)
    path = output / f"克隆试听-{args.quality}-{args.steps}步.wav"
    path.write_bytes(data)
    import numpy as np
    import soundfile as sf
    samples, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 24000:
        raise RuntimeError("Unexpected output sample rate")
    info.update(
        text=text, recognized=engine.transcribe(samples),
        rms=float(np.sqrt(np.mean(samples**2))),
        tail_rms=float(np.sqrt(np.mean(samples[-sample_rate:]**2))),
        peak=float(np.max(np.abs(samples))),
        profile=profile["id"], file=str(path),
    )
    (output / f"实测-{args.quality}-{args.steps}步.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("RESULT", json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
