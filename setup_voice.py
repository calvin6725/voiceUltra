"""Download and verify the official CPU voice models into this project."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RELEASE_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/"


def request(url, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "VoiceLab-local/2", **headers}
        ),
        timeout=60,
    )


def download(tag, name):
    with request(RELEASE_API + tag) as response:
        release = json.load(response)
    asset = next(a for a in release["assets"] if a["name"] == name)
    destination = MODELS / name
    expected = asset.get("digest") or ""
    if destination.exists() and destination.stat().st_size == asset["size"]:
        with destination.open("rb") as existing:
            checksum = hashlib.file_digest(existing, "sha256").hexdigest()
        if not expected or expected == "sha256:" + checksum:
            return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        completed = temporary.stat().st_size if temporary.exists() else 0
        if completed == asset["size"]:
            break
        try:
            headers = {"Accept": "application/octet-stream"}
            if completed:
                headers["Range"] = f"bytes={completed}-"
            with request(
                asset["url"] + f"?download=1&time={time.time_ns()}", **headers
            ) as source:
                if source.status != 206:
                    completed = 0
                with temporary.open("ab" if completed else "wb") as target:
                    last_report = -1
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        completed += len(chunk)
                        percent = completed * 100 // asset["size"]
                        if percent // 10 != last_report:
                            last_report = percent // 10
                            print(f"{name}: {percent}%", flush=True)
        except (OSError, TimeoutError) as exc:
            if attempt == 4:
                raise
            print(f"Retrying {name}: {exc}", flush=True)
    if completed != asset["size"]:
        raise RuntimeError(f"Incomplete download: {name}")
    with temporary.open("rb") as downloaded:
        checksum = hashlib.file_digest(downloaded, "sha256").hexdigest()
    if expected and expected != "sha256:" + checksum:
        raise RuntimeError(f"Checksum mismatch: {name}")
    temporary.replace(destination)
    return destination


def archive(tag, name):
    directory = MODELS / name.removesuffix(".tar.bz2")
    if (directory / ".ready").exists():
        return directory
    source = download(tag, name)
    with tarfile.open(source) as package:
        package.extractall(MODELS, filter="data")
    (directory / ".ready").touch()
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=["fast", "standard", "full"], default="fast")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--speaker", action="store_true")
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    names = {
        "fast": "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
        "standard": "sherpa-onnx-zipvoice-distill-fp32-zh-en-emilia",
        "full": "sherpa-onnx-zipvoice-zh-en-emilia",
    }
    archive("tts-models", names[args.quality] + ".tar.bz2")
    if args.quality == "full":
        archive("tts-models", names["fast"] + ".tar.bz2")
    download("vocoder-models", "vocos_24khz.onnx")
    download("tts-models", "lexicon-en-us.txt")
    if args.asr:
        archive(
            "asr-models",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
        )
    if args.speaker:
        download("speaker-recongition-models", "wespeaker_zh_cnceleb_resnet34.onnx")
    print("Models ready.", flush=True)


if __name__ == "__main__":
    main()
