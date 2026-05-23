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
from pynput import keyboard

_cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yml").read_text())["stt"]
SAMPLE_RATE = _cfg["sample_rate"]
BLOCK_SIZE  = _cfg["block_size"]


def load_model(models_dir: Path, model_id: str) -> nn.Module:
    local_path = models_dir / model_id.split("/")[-1]
    if not local_path.exists():
        print(f"[STT] Downloading {model_id} to {local_path}")
        snapshot_download(repo_id=model_id, local_dir=str(local_path))
    print(f"[STT] Loading from {local_path}")
    model = load_stt(str(local_path))
    print("[STT] Ready.")
    return model


def transcribe(model: nn.Module, audio: np.ndarray) -> tuple[str, float, float, float]:
    """Returns (transcript, audio_duration_s, transcription_time_s, rtf)."""
    audio_duration = len(audio) / SAMPLE_RATE
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        t0 = time.time()
        result = model.generate(tmp.name)
        transcription_time = time.time() - t0
    text = result.text if hasattr(result, "text") else str(result)
    rtf = transcription_time / audio_duration if audio_duration > 0 else 0.0
    print(f"[STT] audio={audio_duration:.2f}s  transcription={transcription_time:.2f}s  RTF={rtf:.3f}")
    return text, audio_duration, transcription_time, rtf


def listen_loop(
    model: nn.Module,
    on_transcript: Callable[[str, float, float, float], None],
    stop_event: threading.Event,
) -> None:
    audio_buf: list[np.ndarray] = []
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    _recording = threading.Event()

    def audio_callback(indata: np.ndarray, *_) -> None:
        if _recording.is_set():
            audio_q.put(indata[:, 0].copy())

    def on_press(key):
        if key == keyboard.Key.cmd and not _recording.is_set():
            audio_buf.clear()
            while not audio_q.empty():  # flush stale audio
                audio_q.get_nowait()
            _recording.set()
            print("[STT] Recording... (release Cmd to transcribe)")

    def on_release(key):
        if key == keyboard.Key.cmd and _recording.is_set():
            _recording.clear()
            while not audio_q.empty():
                audio_buf.append(audio_q.get_nowait())
            if not audio_buf:
                return
            audio = np.concatenate(audio_buf)
            audio_buf.clear()
            if len(audio) / SAMPLE_RATE < 0.4:
                print("[STT] Too short, ignoring.")
                return
            text, audio_dur, trans_time, rtf = transcribe(model, audio)
            text = text.strip()
            if text:
                on_transcript(text, audio_dur, trans_time, rtf)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=BLOCK_SIZE, callback=audio_callback,
    ):
        print("[STT] Hold Cmd to speak, release to transcribe.")
        while not stop_event.is_set():
            time.sleep(0.05)

    listener.stop()


def start(
    model: nn.Module,
    on_transcript: Callable[[str, float, float, float], None],
) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    t = threading.Thread(target=listen_loop, args=(model, on_transcript, stop_event), daemon=True)
    t.start()
    return t, stop_event


def stop(stop_event: threading.Event) -> None:
    stop_event.set()
    print("[STT] Stopped.")


if __name__ == "__main__":
    MODEL_ID   = _cfg["model_id"]
    MODELS_DIR = Path(__file__).parent.parent / _cfg["models_dir"]

    model = load_model(models_dir=MODELS_DIR, model_id=MODEL_ID)
    thread, stop_event = start(model, lambda text, audio_dur, trans_time, rtf: print(f"\n[STT transcript] {text}  (RTF={rtf:.3f})\n"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop(stop_event)
