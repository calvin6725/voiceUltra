import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from voice_engine import EngineBusy, ROOT, VoiceEngine

ENGINE = VoiceEngine()
STATIC = {
    "index.html", "app.js", "styles.css", "voice-lab.css", "clone.js",
    "video.html", "video.js", "video.css", "tts.css", "overlay.css",
    "vendor/lucide.min.js",
}


class VoiceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_bytes(self, status, content_type, data, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, payload):
        self.send_bytes(status, "application/json; charset=utf-8",
                        json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def read_body(self, maximum=262144):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= maximum:
            raise ValueError("请求为空或文件过大")
        return self.rfile.read(size)

    def read_json(self):
        return json.loads(self.read_body().decode("utf-8"))

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == "/api/clone/status":
                self.send_json(200, ENGINE.status())
                return
            if url.path.startswith("/api/clone/reference/"):
                reference_id = url.path.rsplit("/", 1)[-1]
                ENGINE.reference(reference_id)
                data = ENGINE._path(reference_id).with_suffix(".wav").read_bytes()
                self.send_bytes(200, "audio/wav", data)
                return
            if url.path == "/api/voices":
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                     "-File", str(ROOT / "tts_helper.ps1"), "-Mode", "voices"],
                    capture_output=True, timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                voices = []
                for line in result.stdout.decode("utf-8", "ignore").splitlines():
                    name, _, language = line.strip().partition("|")
                    if name:
                        voices.append({"name": name, "language": language})
                self.send_json(200, {"available": bool(voices), "voices": voices})
                return
            relative = url.path.lstrip("/") or "index.html"
            if relative not in STATIC:
                self.send_json(404, {"error": "File not found"})
                return
            self.send_bytes(200, self.guess_type(relative),
                            (ROOT / relative).read_bytes())
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_POST(self):
        try:
            url = urlsplit(self.path)
            if url.path == "/api/clone/reference":
                query = parse_qs(url.query)
                item = ENGINE.import_reference(
                    self.read_body(100 * 1024 * 1024),
                    query.get("name", ["参考录音"])[0],
                    float(query.get("start", ["0"])[0]),
                    float(query.get("duration", ["12"])[0]),
                )
                self.send_json(200, item)
                return
            if url.path == "/api/clone/profile":
                payload = self.read_json()
                item = ENGINE.save_profile(
                    payload.get("id"), payload.get("name", ""),
                    payload.get("text", ""),
                )
                self.send_json(200, item)
                return
            if url.path == "/api/clone/generate":
                payload = self.read_json()
                data, info = ENGINE.generate(
                    payload.get("id"), payload.get("text", ""),
                    payload.get("quality", "fast"), float(payload.get("speed", 1)),
                    int(payload.get("steps", 4)),
                )
                self.send_bytes(200, "audio/wav", data, {
                    "X-Elapsed-Seconds": info["elapsed"],
                    "X-Audio-Seconds": info["duration"],
                })
                return
            if url.path == "/api/tts":
                payload = self.read_json()
                text = str(payload.get("text", "")).strip()
                if not text or len(text) > 12000:
                    raise ValueError("文本为空或超过 12000 字")
                voice = str(payload.get("voice", "Microsoft Huihui Desktop"))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "speech.wav"
                    result = subprocess.run([
                        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(ROOT / "tts_helper.ps1"), "-Mode", "speak",
                        "-TextBase64", base64.b64encode(text.encode()).decode(),
                        "-VoiceBase64", base64.b64encode(voice.encode()).decode(),
                        "-OutputPath", str(path),
                    ], capture_output=True, timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode or not path.is_file():
                        raise RuntimeError("Windows 本机语音合成失败")
                    self.send_bytes(200, "audio/wav", path.read_bytes())
                return
            self.send_json(404, {"error": "Unknown endpoint"})
        except EngineBusy as exc:
            self.send_json(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_DELETE(self):
        url = urlsplit(self.path)
        if not url.path.startswith("/api/clone/profile/"):
            self.send_json(404, {"error": "Unknown endpoint"})
            return
        try:
            ENGINE.delete_profile(url.path.rsplit("/", 1)[-1])
            self.send_json(200, {"deleted": True})
        except EngineBusy as exc:
            self.send_json(409, {"error": str(exc)})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"Voice Lab: http://localhost:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), VoiceHandler).serve_forever()
