import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import psutil
import websockets
import yaml

import llm
import stt
import tts
from track_metrics import MetricsTracker

CLIENTS: set = set()
_proc = psutil.Process(os.getpid())


def _ram() -> float:
    return _proc.memory_info().rss / 1024 ** 3


async def broadcast(msg: dict) -> None:
    if not CLIENTS:
        return
    data = json.dumps(msg, ensure_ascii=False)
    await asyncio.gather(*[c.send(data) for c in CLIENTS], return_exceptions=True)


async def ws_handler(ws) -> None:
    CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())

    stt_cfg = cfg["stt"]
    llm_cfg = cfg["llm"]
    tts_cfg = cfg["tts"]

    tracker = MetricsTracker(ROOT / "logs")

    print("[Main] Loading models...")
    t0 = time.time()
    stt_model = stt.load_model(models_dir=ROOT / stt_cfg["models_dir"], model_id=stt_cfg["model_id"])
    tracker.log_model_load("stt", time.time() - t0)

    t0 = time.time()
    llm_model, tokenizer = llm.load_model(models_dir=ROOT / llm_cfg["models_dir"], model_id=llm_cfg["model_id"])
    tracker.log_model_load("llm", time.time() - t0)

    t0 = time.time()
    tts_model = tts.load_model(models_dir=ROOT / tts_cfg["models_dir"], language=tts_cfg["language"], model_url=tts_cfg["model_url"])
    tracker.log_model_load("tts", time.time() - t0)

    system_prompt = (ROOT / llm_cfg["system_prompt"]).read_text()
    is_speaking = False

    loop = asyncio.new_event_loop()

    def send(msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)

    def on_transcript(text: str, audio_dur: float, trans_time: float, rtf: float) -> None:
        nonlocal is_speaking
        if is_speaking:
            tracker.log_dropped_utterance()
            return

        turn_start = time.time()
        print(f"\n[You]     {text}")
        print(f"[STT]     RTF={rtf:.3f}  audio={audio_dur:.2f}s  transcription={trans_time:.2f}s")
        print(f"[RAM]     idle → {_ram():.2f} GB")
        tracker.log_stt(audio_dur, trans_time, rtf, text)
        send({"type": "transcript", "text": text})

        response, in_tok, out_tok, gen_time, truncated = llm.generate_response(
            llm_model, tokenizer, text, system_prompt, llm_cfg["max_tokens"]
        )
        print(f"[Motiram] {response}")
        print(f"[LLM]     {out_tok} tokens  {out_tok/gen_time:.1f} tok/s  {'[TRUNCATED] ' if truncated else ''}({gen_time:.1f}s)")
        print(f"[RAM]     after LLM → {_ram():.2f} GB")
        tracker.log_llm(in_tok, out_tok, gen_time, truncated)
        send({"type": "response", "text": response})

        is_speaking = True
        send({"type": "status", "state": "speaking"})
        audio_dur_tts, synth_time, in_chars, clean_chars, txt_truncated = tts.speak(
            tts_model, response, tts_cfg["speaker"], tts_cfg["sample_rate"]
        )
        print(f"[TTS]     RTF={synth_time/audio_dur_tts:.3f}  audio={audio_dur_tts:.2f}s  synth={synth_time:.2f}s")
        print(f"[RAM]     after TTS → {_ram():.2f} GB\n")
        tracker.log_tts(in_chars, clean_chars, synth_time, audio_dur_tts, txt_truncated)

        total_latency = time.time() - turn_start
        tracker.log_turn(trans_time, gen_time, synth_time, total_latency, _ram())

        is_speaking = False
        send({"type": "done"})

    async def run_ws_server() -> None:
        async with websockets.serve(ws_handler, "localhost", 8765):
            await asyncio.Future()

    def start_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_ws_server())

    ws_thread = threading.Thread(target=start_loop, daemon=True)
    ws_thread.start()

    print(f"[Main] All models loaded. RAM (idle): {_ram():.2f} GB\n")
    print("Open index.html in your browser, or speak directly.")
    print("Ctrl+C to quit.\n")

    webbrowser.open(str(ROOT / "frontend" / "index.html"))

    thread, stop_event = stt.start(stt_model, on_transcript)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stt.stop(stop_event)
        tracker.close()
        print("[Main] Bye!")


if __name__ == "__main__":
    main()
