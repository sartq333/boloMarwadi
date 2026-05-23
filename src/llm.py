from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from mlx_lm import generate, load

def load_model(models_dir: Path, model_id: str) -> tuple[Any, Any]:
    local_path = models_dir / model_id.split("/")[-1]
    if not local_path.exists():
        print(f"[LLM] Downloading {model_id} to {local_path}")
        snapshot_download(repo_id=model_id, local_dir=str(local_path))
    print(f"[LLM] Loading from {local_path}")
    model, tokenizer = load(str(local_path))
    print("[LLM] Ready.")
    return model, tokenizer

def generate_response(
    model: Any,
    tokenizer: Any,
    user_text: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


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
            response = generate_response(model, tokenizer, user_input, SYSTEM_PROMPT, MAX_TOKENS)
            print(f"Motiram: {response}\n")
        except KeyboardInterrupt:
            print("\n[LLM] Bye!")
            break
