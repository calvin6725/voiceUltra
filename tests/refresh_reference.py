import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_engine import VoiceEngine
import soundfile as sf

engine = VoiceEngine()
profile = next(p for p in engine.profiles() if p["name"] == "素材原声")
samples, warnings, peak, rms = engine._decode(
    Path(r"C:\Users\12495\Desktop\素材.mp3").read_bytes(), 0, 12
)
sf.write(str(engine._path(profile["id"]).with_suffix(".wav")),
         samples, 24000, subtype="PCM_16")
profile.update(warnings=warnings, peak=peak, rms=rms)
engine._save(profile)
print(json.dumps(profile, ensure_ascii=True))
