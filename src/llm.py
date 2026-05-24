import re
import time
from pathlib import Path
from typing import Any, Generator

import yaml
from huggingface_hub import snapshot_download
from mlx_lm import generate, load, stream_generate

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?।])\s+')


def load_model(models_dir: Path, model_id: str) -> tuple[Any, Any]:
    local_path = models_dir / model_id.split("/")[-1]
    if not local_path.exists():
        print(f"[LLM] Downloading {model_id} to {local_path}")
        snapshot_download(repo_id=model_id, local_dir=str(local_path))
    print(f"[LLM] Loading from {local_path}")
    model, tokenizer = load(str(local_path))
    print("[LLM] Ready.")
    return model, tokenizer


def stream_sentences(
    model: Any,
    tokenizer: Any,
    user_text: str,
    system_prompt: str,
    max_tokens: int,
) -> Generator[tuple[str, dict | None], None, None]:
    """
    Yields (sentence, stats) pairs.
    stats is None for intermediate sentences; populated on the final yield:
      {full_text, input_tokens, output_tokens, gen_time, truncated}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_tokens = len(tokenizer.encode(prompt))

    t0 = time.time()
    buffer = ""
    full_text = ""
    output_tokens = 0

    for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        if not chunk.text:
            continue
        buffer += chunk.text
        full_text += chunk.text
        output_tokens += 1

        parts = _SENTENCE_SPLIT.split(buffer)
        for sentence in parts[:-1]:
            if sentence.strip():
                yield sentence.strip(), None
        buffer = parts[-1]

    gen_time = time.time() - t0
    stats = {
        "full_text": full_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gen_time": gen_time,
        "truncated": output_tokens >= max_tokens,
    }
    yield buffer.strip(), stats


def generate_response(
    model: Any,
    tokenizer: Any,
    user_text: str,
    system_prompt: str,
    max_tokens: int,
) -> tuple[str, int, int, float, bool]:
    """Returns (text, input_tokens, output_tokens, generation_time_s, truncated)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    t0 = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    gen_time = time.time() - t0
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(response))
    truncated = output_tokens >= max_tokens
    return response, input_tokens, output_tokens, gen_time, truncated


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((ROOT / "config.yml").read_text())["llm"]

    MODELS_DIR    = ROOT / cfg["models_dir"]
    MODEL_ID      = cfg["model_id"]
    MAX_TOKENS    = cfg["max_tokens"]
    SYSTEM_PROMPT = (ROOT / cfg["system_prompt"]).read_text()

    model, tokenizer = load_model(MODELS_DIR, MODEL_ID)

    print("\nType something in Hindi or English. Ctrl+C to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response, _, _, gen_time, truncated = generate_response(model, tokenizer, user_input, SYSTEM_PROMPT, MAX_TOKENS)
            print(f"Motiram: {response}  ({gen_time:.1f}s{'  [TRUNCATED]' if truncated else ''})\n")
        except KeyboardInterrupt:
            print("\n[LLM] Bye!")
            break
