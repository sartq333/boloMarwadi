import json
import time
from pathlib import Path

class MetricsTracker:
    def __init__(self, logs_dir: Path) -> None:
        logs_dir.mkdir(exist_ok=True)
        self.log_path = logs_dir / f"session_{int(time.time())}.log"
        self._session_start = time.time()
        self._utterance_count = 0
        self._dropped_count = 0
        self._write({"event": "session_start", "ts": self._session_start})
        print(f"[Metrics] Logging to {self.log_path}")

    def _write(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_model_load(self, component: str, load_time_s: float) -> None:
        self._write({
            "event": "model_load", "ts": time.time(),
            "component": component, "load_time_s": round(load_time_s, 3),
        })

    def log_stt(self, audio_duration_s: float, transcription_time_s: float,
                rtf: float, transcript: str) -> None:
        self._utterance_count += 1
        self._write({
            "event": "stt", "ts": time.time(),
            "utterance_index": self._utterance_count,
            "audio_duration_s": round(audio_duration_s, 3),
            "transcription_time_s": round(transcription_time_s, 3),
            "rtf": round(rtf, 3),
            "transcript_chars": len(transcript),
        })

    def log_llm(self, input_tokens: int, output_tokens: int,
                generation_time_s: float, truncated: bool) -> None:
        tps = round(output_tokens / generation_time_s, 1) if generation_time_s > 0 else 0.0
        self._write({
            "event": "llm", "ts": time.time(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tps,
            "generation_time_s": round(generation_time_s, 3),
            "truncated": truncated,
        })

    def log_tts(self, input_chars: int, cleaned_chars: int, synthesis_time_s: float,
                audio_duration_s: float, text_truncated: bool) -> None:
        rtf = round(synthesis_time_s / audio_duration_s, 3) if audio_duration_s > 0 else 0.0
        cps = round(cleaned_chars / synthesis_time_s, 1) if synthesis_time_s > 0 else 0.0
        self._write({
            "event": "tts", "ts": time.time(),
            "input_chars": input_chars,
            "cleaned_chars": cleaned_chars,
            "synthesis_time_s": round(synthesis_time_s, 3),
            "audio_duration_s": round(audio_duration_s, 3),
            "rtf": rtf,
            "chars_per_second": cps,
            "text_truncated": text_truncated,
        })

    def log_turn(self, stt_time_s: float, llm_time_s: float,
                 tts_time_s: float, total_latency_s: float, memory_gb: float) -> None:
        self._write({
            "event": "turn", "ts": time.time(),
            "stt_time_s": round(stt_time_s, 3),
            "llm_time_s": round(llm_time_s, 3),
            "tts_time_s": round(tts_time_s, 3),
            "total_latency_s": round(total_latency_s, 3),
            "memory_gb": round(memory_gb, 2),
        })

    def log_dropped_utterance(self) -> None:
        self._dropped_count += 1
        self._write({
            "event": "dropped_utterance", "ts": time.time(),
            "total_dropped": self._dropped_count,
        })

    def close(self) -> None:
        self._write({
            "event": "session_end", "ts": time.time(),
            "duration_s": round(time.time() - self._session_start, 1),
            "total_utterances": self._utterance_count,
            "total_dropped": self._dropped_count,
        })
        print(f"[Metrics] Session complete — {self.log_path}")
