# voiceUltra — 本地 AI 声音克隆

适配 **Intel Core Ultra 5 125H + 约 32GB 内存 + Intel Arc 集成显卡**。
本地运行，不上传云端。引擎：ZipVoice（sherpa-onnx）+ SenseVoice 识别。

## 为什么不“假”

ZipVoice 本身是 SOTA 零样本克隆（123M 参数，UTMOS 与 F5-TTS 相当）。
“机器人感”通常来自外围环节，本项目已逐项修复：

| 假感来源 | 本项目的做法 |
|---|---|
| INT8 蒸馏 + 4 步采样太糙 | 默认 **自然优先（full，16 步）/ 均衡（standard，8 步）**，快速版只做试听 |
| 参考录音削波、底噪、音量不稳 | 自动**削波修复 + 高通 + 降噪 + 响度归一**，并给出 0–100 参考质量分 |
| 原话识别错字 | 识别后必须**人工核对原话**才能保存 |
| 长文整段合成，语调平、停顿齐 | 按句**分段合成**，标点驱动自然停顿 + 每段 ±2% 语速微扰 |
| 声码器输出偏薄偏平 | 默认**后期增强**（厚度/存在感 EQ + 轻压缩 + 响度统一） |

## 硬件适配

- **CPU**：Ultra 5 125H 是 2P+8E+2LPE，ONNX 全核反而更慢。程序自动选
  **6 线程**（P 核 + 少量 E 核），可用 `VOICE_THREADS` 覆盖。
- **Arc 核显**：装 `onnxruntime-directml` 后自动走 **DirectML** 用核显加速；
  也有 OpenVINO 路径。不装则自动回退 CPU，功能不变。
- **内存**：32GB 远超 ZipVoice 需要；同一时间只驻留一个模型。
- 状态接口 `/api/clone/status` 会返回本机 CPU/GPU/线程/provider。

## 快速开始（Windows）

```powershell
# 1. 安装依赖（项目目录下）
.runtime\Scripts\python.exe -m pip install -r requirements-voice.txt
# 可选：Arc 核显加速
.runtime\Scripts\python.exe -m pip install onnxruntime-directml

# 2. 下载模型（默认 standard；想要最像用 --quality full 或 --all）
.runtime\Scripts\python.exe setup_voice.py --quality standard --asr
# 或全部下载：
.runtime\Scripts\python.exe setup_voice.py --all --asr

# 3. 双击“启动声线实验室.bat”打开 http://localhost:8001/
```

## 参考录音怎么录（最重要）

1. **6–15 秒**，单人、安静、无伴奏、无混响，完整句子。
2. 离麦克风 15–30cm，峰值别爆红（程序会提示削波）。
3. 提取后**试听参考原声**，核对自动识别的原话（人名/口语/英文最易错）。
4. 看到“参考质量 xx 分”，低于 60 分建议重录。
5. 保存音色后，用“自然优先 + 高精细（16 步）”生成试听。

## 模型与精度

- **自然优先（full）**：未蒸馏完整模型，16 步，最像、最慢。
- **均衡（standard）**：FP32 蒸馏，8 步，速度与音质平衡（默认）。
- **快速试听（fast）**：INT8 蒸馏，4 步，只用于快速试听。
- 语速 0.75–1.35 倍；停顿强度 0.5–2 倍；后期增强默认开。
- 长文自动按句分段；单次最多 3000 字。

## 文件说明

- `voice_engine.py` 克隆引擎（分段合成 + 清洗 + 母带）
- `audio_dsp.py` 参考清洗/评分/后期（纯 numpy/scipy）
- `text_frontend.py` 文本规范化/分句/停顿
- `hardware.py` CPU/GPU/provider 自适应
- `voice_server.py` 本地 HTTP 服务（127.0.0.1:8001）
- `setup_voice.py` 模型下载（校验 SHA256）
- `requirements-voice.txt` 基础依赖；`requirements-arc.txt` 核显加速（可选）

## 隐私

参考录音与音色保存在本机 `voice-data/`，不上传。模型与运行环境均在
项目文件夹内，安装后可离线使用。
