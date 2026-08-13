# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LiveTalking is a real-time streaming digital-human engine. It takes text or audio input, optionally runs it through an LLM, synthesizes speech via TTS, and renders lip-sync video through deep-learning avatar models (Wav2Lip, MuseTalk, Ultralight). Output can be WebRTC, RTMP, RTCPush, or virtual camera.

The upstream project is CUDA-based (PyTorch + torch.cuda). This fork is being adapted to run on **Ascend 910B (NPU)**, so device assumptions need to be generalized rather than hard-coded to CUDA.

## Common commands

### Environment setup

#### CUDA / local development

```bash
conda create -n livetalking python=3.12
conda activate livetalking
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

#### Ascend 910B (target deployment)

Use the Ascend PyTorch container; it already provides CANN and `torch_npu`:

```bash
docker run -itd --privileged --net=host --ipc=host --name=ascend-livetalking \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/:/usr/local/sbin:ro \
  -v /data/LiveTalking:/data/LiveTalking \
  -p 8188:8188 \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-pytorch:24.0.0-A2-2.1.0-ubuntu20.04 \
  /bin/bash
```

Inside the container:

```bash
docker exec -it ascend-livetalking bash
cd /data/LiveTalking
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirements.txt
# The Ascend 24.0.0-A2-2.1.0 container ships PyTorch 2.1.0, which is not
# compatible with NumPy 2.x or very recent transformers. Pin these:
pip install numpy==1.26.4 transformers==4.35.2
```

Do **not** install CUDA PyTorch wheels inside the Ascend container.

### Model and avatar setup

```bash
# Place the Wav2Lip checkpoint
mkdir -p models
cp <download>/wav2lip256.pth models/wav2lip.pth

# Place the default avatar
mkdir -p data/avatars
tar -xzf <download>/wav2lip256_avatar1.tar.gz -C data/avatars/
```

### Run the server

```bash
# Default WebRTC mode (auto-detects device)
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1

# Force Ascend NPU (source CANN environment first)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --device npu

# Use YAML config instead of CLI args
python app.py --config config.yaml
```

The server listens on TCP `8010` by default (or `listenport` from config). WebRTC also requires UDP ports `1-65536` to be open.

### Useful development invocations

```bash
# List available TTS voices / reference voices for EdgeTTS
python -c "import edge_tts; print(edge_tts.list_voices())"

# Run the benchmark/ASR tool
python benchmark_asr.py

# Check which device the project would select
python -c "from utils.device import initialize_device; print(initialize_device())"
```

> There are currently no automated tests, no lint configuration, and no Makefile/pyproject.toml. Validation is done by running `app.py` and exercising the API or browser UI.

## High-level architecture

### Entry point and server

`app.py` is the only entry point. It:

1. Parses options via `config.py` (CLI args override `config.yaml`).
2. Imports the selected avatar module (`avatars/musetalk_avatar.py`, `avatars/wav2lip_avatar.py`, or `avatars/ultralight_avatar.py`), which registers the avatar class through the plugin registry.
3. Loads the model and the default avatar into `global_avatars`.
4. Starts an `aiohttp` server and wires routes from `server/routes.py` plus WebRTC handling in `server/rtc_manager.py`.

Static files are served from `web/`. Key pages: `index.html` (WebRTC client), `avatar.html` (avatar creator), `admin.html` (monitoring).

### Plugin registry

`registry.py` provides a decorator-based registry with categories:

- `stt` — speech-to-text (local ASR endpoint in `server/asr_server.py`)
- `llm` — LLM integration (`llm.py`)
- `tts` — text-to-speech (`tts/`)
- `avatar` — avatar/rendering engines (`avatars/*_avatar.py`)
- `output` / `streamout` — transport outputs (`streamout/`)

Classes self-register when their module is imported. To add a new TTS, avatar, or output transport, create a subclass of the corresponding base class, decorate it with `@register("<category>", "<name>")`, and ensure its module is imported in `BaseAvatar` or `app.py`.

### Per-session pipeline

`BaseAvatar` (in `avatars/base_avatar.py`) defines the runtime pipeline for one user session:

```
TTS → ASR/audio feature extractor → inference batch → paste-back / post-process → output transport
```

- `tts`: enqueues text, produces audio chunks.
- `asr` (misnamed; it is the audio-feature module): consumes audio, produces feature batches (e.g., Mel spectrograms for Wav2Lip, Whisper features for MuseTalk).
- `inference()`: pulls a feature batch, runs `inference_batch()`, and puts predicted mouth frames into `res_frame_queue`.
- `process_frames()`: composites frames, adds the watermark, pushes video/audio to the selected output, and optionally records to `data/record/{sessionid}.mp4`.

Each session is managed by the singleton `SessionManager` (`server/session_manager.py`), which enforces `max_session`. WebRTC connections are handled by `RTCManager` (`server/rtc_manager.py`), which creates a session per offer and tears it down on disconnect.

### Model-specific avatars

Each avatar lives in its own sub-package under `avatars/`:

- `wav2lip_avatar.py`: loads `models/wav2lip.pth`, uses `MelASR`, expects `data/avatars/{avatar_id}/{full_imgs,face_imgs,coords.pkl}`.
- `musetalk_avatar.py`: uses Whisper features and a diffusion UNet; avatar assets are generated differently.
- `ultralight_avatar.py`: uses a lightweight UNet with HuBERT audio features.

Common utilities are in `utils/` (`device.py`, `image.py`, `audio.py`, `logger.py`).

### Transports / outputs

Output implementations live in `streamout/` and inherit from `BaseOutput`:

- `webrtc`: sends A/V through `aiortc` tracks (`server/webrtc.py` wraps the avatar as a `HumanPlayer`).
- `rtmp`: uses FFmpeg to push to an RTMP endpoint.
- `virtualcam`: renders to a virtual camera device.
- `rtcpush`: WHIP-style WebRTC push.

### Configuration

`config.yaml` holds defaults; CLI arguments override it. Important keys:

- `model`: `wav2lip` / `musetalk` / `ultralight`
- `avatar_id`: folder name under `data/avatars/`
- `transport`: `webrtc` / `rtmp` / `virtualcam` / `rtcpush`
- `tts`: `edgetts`, `gpt-sovits`, `cosyvoice`, etc.
- `max_session`, `listenport`, `stun`

API keys for cloud TTS providers go in `.env` (see `.env.example`).

## Important conventions

- **Device handling is not fully centralized.** `utils/device.py` returns `cuda`, `mps`, or `cpu`, but many files still check `torch.cuda.is_available()` directly or call `.cuda()`. For Ascend 910B support, device selection and tensor movement need to be generalized.
- **No unit tests.** Verify changes by starting the server and using the browser UI at `/index.html` or the HTTP API documented in `docs/api.md`.
- **Avatar assets are preprocessed offline.** Do not commit large model weights or avatar image folders. The repository tracks only the code; models go in `models/`, avatars in `data/avatars/`.
- **Multiprocessing start method:** `app.py` sets `mp.set_start_method('spawn')` at module load.
- **Recording temp files:** `BaseAvatar` writes `temp{sessionid}.mp4` and `temp{sessionid}.aac` in the working directory and then muxes them into `data/record/{sessionid}.mp4`.
- **Frame rate assumption:** The engine targets 25 FPS. Audio is processed in 20 ms chunks (`chunk = sample_rate // (fps * 2)`).

## Porting notes for Ascend 910B

- Replace CUDA-only installs with the matching CANN + `torch_npu` wheel.
- Audit all `.cuda()`, `torch.cuda.*`, and `device == 'cuda'` checks; route them through `utils/device.py` or a single device abstraction.
- `wav2lip_avatar.py` explicitly branches on `device == 'cuda'` when loading checkpoints; this must be updated for `npu` map_location behavior.
- Check half-precision/float16 usage in `musetalk_avatar.py` and `ultralight_avatar.py`; Ascend may need different dtype handling or explicit conversion operators.
- Verify that `opencv-python-headless`, `aiortc`, and FFmpeg still behave the same way on the NPU host; the video codec negotiation in `server/rtc_manager.py` is unchanged.

## External documentation

- API reference: `docs/api.md`
- Avatar generation API: `docs/avatar_api.md`
- Admin/monitoring API: `docs/admin_api.md`
- Virtual camera guide: `docs/virtualcam_guide.md`
- FAQ: `assets/faq.md`
- Full docs: https://doc.livetalking.ai/docs (external; may require network access)
