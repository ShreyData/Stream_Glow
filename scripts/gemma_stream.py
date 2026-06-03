#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_API_VERSION = "v1beta"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_SYSTEM_PROMPT = """
You are Gemma Glow, a friendly, professional terminal assistant.
Reply in a clean CLI-friendly format:
- Start with the direct answer.
- Use short paragraphs.
- Use headings only when they help.
- Use compact bullets for lists.
- Avoid huge Markdown blocks, tables, and overexplaining.
- Keep code in fenced blocks when code is useful.
""".strip()


def main() -> int:
    load_env(".env")

    try:
        request = json.load(sys.stdin)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        key_error = validate_api_key(api_key)
        if key_error:
            emit("error", message=key_error)
            return 1

        model = request.get("model") or os.getenv("GEMMA_MODEL") or DEFAULT_MODEL
        payload = build_payload(request)
        stream(model, api_key, payload)
        emit("done")
        return 0
    except Exception as exc:
        emit("error", message=str(exc))
        return 1


def build_payload(request: dict) -> dict:
    contents = []
    for message in request.get("messages", []):
        role = message.get("role", "user")
        if role == "assistant":
            role = "model"
        contents.append(
            {
                "role": role,
                "parts": [{"text": message.get("content", "")}],
            }
        )

    generation_config = {
        "temperature": float(request.get("temperature", 0.7)),
    }

    if request.get("thinking", True):
        generation_config["thinkingConfig"] = {
            "includeThoughts": True,
        }

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    system_prompt = os.getenv("GEMMA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    return payload


def stream(model: str, api_key: str, payload: dict) -> None:
    api_version = os.getenv("GEMINI_API_VERSION", DEFAULT_API_VERSION)
    base_url = os.getenv("GEMINI_API_BASE", DEFAULT_BASE_URL).rstrip("/")
    model_path = urllib.parse.quote(model, safe="")
    url = (
        f"{base_url}/{api_version}/models/{model_path}:"
        f"streamGenerateContent?alt=sse"
    )

    body = json.dumps(payload).encode("utf-8")
    http_request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(http_request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                handle_chunk(json.loads(data))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(format_google_error(exc.code, detail)) from exc


def handle_chunk(chunk: dict) -> None:
    for candidate in chunk.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if not text:
                continue
            if part.get("thought"):
                emit("thought", text=text)
            else:
                emit("answer", text=text)


def validate_api_key(api_key):
    if not api_key:
        return "Set GEMINI_API_KEY or GOOGLE_API_KEY in .env"

    placeholder_values = {
        "your_google_ai_studio_api_key_here",
        "your_api_key_here",
        "paste_your_key_here",
    }
    if api_key.strip() in placeholder_values:
        return "Replace the placeholder GEMINI_API_KEY in .env with a real Google AI Studio API key"

    if len(api_key.strip()) < 20:
        return "GEMINI_API_KEY in .env looks too short; paste the full Google AI Studio API key"

    return None


def format_google_error(status_code: int, detail: str) -> str:
    try:
        payload = json.loads(detail)
        error = payload.get("error", {})
        reason = ""
        for item in error.get("details", []):
            if item.get("reason"):
                reason = item["reason"]
                break

        if reason == "API_KEY_INVALID":
            return (
                "Google rejected GEMINI_API_KEY as invalid. "
                "Create/copy a fresh key from Google AI Studio, paste it into .env, "
                "and make sure there are no spaces or quotes around it."
            )

        message = error.get("message")
        if message:
            return f"Google API HTTP {status_code}: {message}"
    except json.JSONDecodeError:
        pass

    return f"Google API HTTP {status_code}: {detail}"


def load_env(path: str) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def emit(event_type: str, **fields: str) -> None:
    fields["type"] = event_type
    print(json.dumps(fields, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
