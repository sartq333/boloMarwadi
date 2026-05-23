import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml
from TTS.utils.synthesizer import Synthesizer

# match the last sentence-ending punctuation (handles LLM mid-word cutoff)
_LAST_SENTENCE_END = re.compile(r'^(.*[.!?।])', re.DOTALL)


def _clean_for_tts(text: str) -> str:
    text = re.sub(r'[*#_`~]+', '', text)   # strip markdown
    text = text.replace('।', '.')           # danda → period so sentence splitter works
    m = _LAST_SENTENCE_END.match(text)
    if m:
        text = m.group(1)
    return text.strip()


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "download.zip"
    print(f"[TTS] Downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def load_model(models_dir: Path, language: str, model_url: str) -> Synthesizer:
    lang_dir = models_dir / language
    if not lang_dir.exists():
        print(f"[TTS] Downloading {language} model (~1.5 GB)")
        download_and_extract(model_url, models_dir)
    print(f"[TTS] Loading from {lang_dir}")
    model = Synthesizer(
        tts_checkpoint=str(lang_dir / "fastpitch" / "best_model.pth"),
        tts_config_path=str(lang_dir / "fastpitch" / "config.json"),
        vocoder_checkpoint=str(lang_dir / "hifigan" / "best_model.pth"),
        vocoder_config=str(lang_dir / "hifigan" / "config.json"),
    )
    print("[TTS] Ready.")
    return model


def synthesize(model: Synthesizer, text: str, speaker: str) -> tuple[np.ndarray, int, int, bool]:
    """Returns (audio, input_chars, cleaned_chars, text_truncated)."""
    input_len = len(text)
    cleaned = _clean_for_tts(text)
    cleaned_len = len(cleaned)
    text_truncated = cleaned_len < input_len
    if cleaned_len < 3:
        return np.zeros(1, dtype=np.float32), input_len, cleaned_len, text_truncated
    wav = model.tts(cleaned, speaker_name=speaker)
    return np.array(wav, dtype=np.float32), input_len, cleaned_len, text_truncated


def speak(model: Synthesizer, text: str, speaker: str,
          sample_rate: int) -> tuple[float, float, int, int, bool]:
    """Returns (audio_duration_s, synthesis_time_s, input_chars, cleaned_chars, text_truncated)."""
    t0 = time.time()
    audio, input_len, cleaned_len, text_truncated = synthesize(model, text, speaker)
    synthesis_time = time.time() - t0
    audio_duration = len(audio) / sample_rate
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    return audio_duration, synthesis_time, input_len, cleaned_len, text_truncated


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())["tts"]

    SAMPLE_RATE = cfg["sample_rate"]
    LANGUAGE    = cfg["language"]
    SPEAKER     = cfg["speaker"]
    MODELS_DIR  = ROOT / cfg["models_dir"]
    MODEL_URL   = cfg["model_url"]

    model = load_model(MODELS_DIR, LANGUAGE, MODEL_URL)
    speak(model, "राम राम! म्हारो नाम मोती है, थे कियो छो?", SPEAKER, SAMPLE_RATE)