import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
import mlx.nn as nn
import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from huggingface_hub import snapshot_download
from mlx_audio.stt.utils import load as load_stt

_cfg = yaml.safe_load((Path(__file__).parent / "config.yml").read_text())["stt"]
SAMPLE_RATE           = _cfg["sample_rate"]
BLOCK_SIZE            = _cfg["block_size"]
SILENCE_RMS_THRESHOLD = _cfg["silence_rms_threshold"]
SILENCE_CUTOFF_S      = _cfg["silence_cutoff_s"]
MIN_UTTERANCE_S       = _cfg["min_utterance_s"]


def load_model(models_dir: Path, model_id: str) -> nn.Module:
    local_path = models_dir / model_id.split("/")[-1]
    if not local_path.exists():
        print(f"[STT] Downloading {model_id} to {local_path}")
        snapshot_download(repo_id=model_id, local_dir=str(local_path))
    print(f"[STT] Loading from {local_path}")
    model = load_stt(str(local_path))
    print("[STT] Ready.")
    return model

def transcribe(model: nn.Module, audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        result = model.generate(tmp.name)
    if hasattr(result, "text"):
        return result.text 
    return str(result) 

def listen_loop(model: nn.Module, on_transcript: Callable[[str], None], stop_event: threading.Event) -> None:
    audio_buf: list[np.ndarray] = []
    silence_blocks = 0
    silence_cutoff = int(SILENCE_CUTOFF_S * SAMPLE_RATE / BLOCK_SIZE)
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, *_) -> None:
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=callback):
        print("[STT] Listening... (speak in Hindi or English)")
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speech = float(np.sqrt(np.mean(block ** 2))) > SILENCE_RMS_THRESHOLD

            if is_speech:
                audio_buf.append(block)
                silence_blocks = 0
            elif audio_buf:
                silence_blocks += 1
                audio_buf.append(block)
                if silence_blocks >= silence_cutoff:
                    audio = np.concatenate(audio_buf)
                    audio_buf.clear()
                    silence_blocks = 0
                    if len(audio) / SAMPLE_RATE >= MIN_UTTERANCE_S:
                        text = transcribe(model, audio).strip()
                        if text:
                            on_transcript(text)

def start(model: nn.Module, on_transcript: Callable[[str], None]) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    t = threading.Thread(target=listen_loop, args=(model, on_transcript, stop_event), daemon=True)
    t.start()
    return t, stop_event

def stop(stop_event: threading.Event) -> None:
    stop_event.set()
    print("[STT] Stopped.")

if __name__ == "__main__":
    MODEL_ID   = _cfg["model_id"]
    MODELS_DIR = Path(__file__).parent / _cfg["models_dir"]

    model = load_model(models_dir=MODELS_DIR, model_id=MODEL_ID) 
    thread, stop_event = start(model, lambda text: print(f"\n[STT transcript] {text}\n"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop(stop_event)