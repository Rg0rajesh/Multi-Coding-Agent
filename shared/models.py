"""
Single entry point for talking to Ollama. Every module in the pipeline goes
through query_llm() instead of hitting the Ollama API directly -- that's what
lets us log token counts and latency consistently everywhere, and swap the
backend later without touching pipeline code.
"""
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


class LLMError(Exception):
    """Raised when Ollama can't be reached, times out, or returns an error."""


def query_llm(
    prompt: str,
    model: str = "deepseek-coder:6.7b",
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
    seed: int = 42,
) -> dict:
    """
    Send one prompt to a local Ollama model.

    Returns:
        {
            "generated_text": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "latency_sec": float,
        }
    """
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "seed": seed,
    }
    model_lower = model.lower()
    if "codellama" in model_lower:
        options["num_ctx"] = 2048
        options["stop"] = ["<EOT>", "</s>", "\nif __name__"]
    elif "qwen" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = ["<|im_end|>", "<|endoftext|>", "\nif __name__"]
    elif "llama" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = ["<|eot_id|>", "<|end_of_text|>", "</s>", "\nif __name__"]
    elif "codegemma" in model_lower or "gemma" in model_lower:
        options["num_ctx"] = 2048
        options["stop"] = [
            "<end_of_turn>",
            "<|end_of_turn|>",
            "<start_of_turn>",
            "<eos>",
            "<|file_separator|>",
            "\nif __name__",
        ]
        options["repeat_penalty"] = 1.1
    elif "deepseek" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = ["<｜end of sentence｜>", "<|EOT|>", "\nif __name__"]
    elif "starcoder" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = [
            "<|endoftext|>",
            "<file_sep>",
            "<fim_prefix>",
            "\nif __name__",
        ]
        options["repeat_penalty"] = 1.1
    elif "phi" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = ["<|endoftext|>", "<|end|>", "\nif __name__"]
    elif "mistral" in model_lower:
        options["num_ctx"] = 4096
        options["stop"] = ["</s>", "\nif __name__"]

    if "num_ctx" not in options:
        options["num_ctx"] = 4096
    if "stop" not in options:
        options["stop"] = ["\nif __name__"]

    if temperature == 0.0:
        options["top_k"] = 1
        options["top_p"] = 1.0
    else:
        options["top_p"] = 0.95
        options["top_k"] = 40

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    start = time.time()
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
    else:
        raise LLMError(f"Ollama call failed for model '{model}': {last_exc}") from last_exc

    data = resp.json()
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return {
        "generated_text": data.get("response", "").strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_sec": round(time.time() - start, 3),
    }
