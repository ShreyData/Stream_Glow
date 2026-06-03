# Gemma Glow

Beautiful Rust CLI for chatting with Google AI Studio models, with a tiny Python streaming bridge.

The default model is:

```text
gemma-4-26b-a4b-it
```

The CLI streams chunks live as they arrive and requests thought summaries when the API/model supports them.
It also renders common Markdown live so lists and bold text look clean in the terminal.

## Setup

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

No Python package install is required. The bridge uses Python's standard library.

## Run

Interactive chat:

```bash
cargo run
```

Single prompt:

```bash
cargo run -- --prompt "Explain transformers in 5 lines"
```

Preview the terminal renderer without calling the API:

```bash
cargo run -- --demo
```

Use another model:

```bash
cargo run -- --model gemini-2.5-flash
```

Disable thought summaries if a model rejects them:

```bash
cargo run -- --no-thinking
```

The bridge includes a default terminal-friendly system prompt. Override it from `.env` if you want another response style:

```bash
GEMMA_SYSTEM_PROMPT=Reply with concise, polished terminal-friendly answers.
```

## Chat Commands

```text
/help   show commands
/model  show current model
/clear  clear chat memory
/exit   quit
```

## Notes

Google's Gemini API exposes streaming through `streamGenerateContent?alt=sse`. Thought summaries are requested with `generationConfig.thinkingConfig.includeThoughts`.

If `gemma-4-26b-a4b-it` is not available on your API key, set `GEMMA_MODEL` in `.env` or pass `--model`.
