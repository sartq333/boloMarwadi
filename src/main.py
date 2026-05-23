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

CLIENTS: set = set()
_proc = psutil.Process(os.getpid())


def _ram() -> str:
    gb = _proc.memory_info().rss / 1024 ** 3
    return f"{gb:.2f} GB"


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
    base = ROOT

    print("[Main] Loading models...")
    stt_model = stt.load_model(
        models_dir=base / stt_cfg["models_dir"],
        model_id=stt_cfg["model_id"],
    )
    llm_model, tokenizer = llm.load_model(
        models_dir=base / llm_cfg["models_dir"],
        model_id=llm_cfg["model_id"],
    )
    tts_model = tts.load_model(
        models_dir=base / tts_cfg["models_dir"],
        language=tts_cfg["language"],
        model_url=tts_cfg["model_url"],
    )

    system_prompt = (base / llm_cfg["system_prompt"]).read_text()
    is_speaking = False

    # event loop for WebSocket broadcasts (runs in its own thread)
    loop = asyncio.new_event_loop()

    def send(msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)

    def on_transcript(text: str, rtf: float) -> None:
        nonlocal is_speaking
        if is_speaking:
            return
        print(f"\n[You]     {text}")
        print(f"[STT]     RTF={rtf:.3f}")
        print(f"[RAM]     idle      → {_ram()}")
        send({"type": "transcript", "text": text})

        t0 = time.time()
        response = llm.generate_response(
            llm_model, tokenizer, text, system_prompt, llm_cfg["max_tokens"]
        )
        print(f"[Motiram] {response}")
        print(f"[RAM]     after LLM → {_ram()}  ({time.time()-t0:.1f}s)")
        send({"type": "response", "text": response})

        is_speaking = True
        send({"type": "status", "state": "speaking"})
        t1 = time.time()
        tts.speak(tts_model, response, tts_cfg["speaker"], tts_cfg["sample_rate"])
        print(f"[RAM]     after TTS → {_ram()}  ({time.time()-t1:.1f}s)\n")
        is_speaking = False
        send({"type": "done"})

    async def run_ws_server() -> None:
        async with websockets.serve(ws_handler, "localhost", 8765):
            await asyncio.Future()  # run forever

    def start_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_ws_server())

    ws_thread = threading.Thread(target=start_loop, daemon=True)
    ws_thread.start()

    print(f"[Main] All models loaded. RAM (idle): {_ram()}\n")
    print("Open index.html in your browser, or speak directly.")
    print("Ctrl+C to quit.\n")

    # open UI automatically
    webbrowser.open(str(base / "frontend" / "index.html"))

    thread, stop_event = stt.start(stt_model, on_transcript)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stt.stop(stop_event)
        print("[Main] Bye!")


if __name__ == "__main__":
    main()