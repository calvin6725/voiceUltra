import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_engine import MODELS
import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "试听样本"
    files = list(output.glob("克隆试听*.wav"))
    report = json.loads(next(output.glob("实测-fast*.json")).read_text(encoding="utf-8"))
    request = urllib.request.Request(
        f"http://localhost:{args.port}/api/tts",
        data=json.dumps({"text": report["text"], "voice": "Microsoft Huihui Desktop"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        baseline = output / "旧版系统朗读对照.wav"
        baseline.write_bytes(response.read())
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(MODELS / "wespeaker_zh_cnceleb_resnet34.onnx"),
        num_threads=4, provider="cpu",
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embedding(path):
        samples, rate = sf.read(path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        from math import gcd
        factor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // factor, rate // factor).astype(np.float32)
        stream = extractor.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        if not extractor.is_ready(stream):
            raise RuntimeError("Reference too short")
        vector = np.asarray(extractor.compute(stream))
        return vector / np.linalg.norm(vector)

    reference = embedding(output / "参考原声.wav")
    scores = {
        path.name: round(float(reference @ embedding(path)), 4)
        for path in [baseline, *files]
    }
    result = {
        "model": "wespeaker_zh_cnceleb_resnet34",
        "metric": "cosine_similarity_not_percentage",
        "scores": scores,
    }
    (output / "声纹对比.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
