# Hybrid LLM Router

Routes each prompt to a **local LLM** (Qwen3 1.7B via Ollama) or a **cloud LLM** (OpenAI) *before* inference,
to save cloud tokens while keeping answer quality acceptable. If the local model fails in Automatic mode,
the request is retried **once** in the cloud.

```
Streamlit UI → Request Processor → Routing Engine ─┬→ Local Engine (Ollama / Qwen3)
                                                   └→ Cloud Engine (OpenAI)
                                     → Response Handler (validation + bounded fallback) → UI
```

## Setup

1. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Ollama (https://ollama.com/download), then pull the model:
   ```bash
   ollama pull qwen3:1.7b
   ```
3. Configure the cloud model (optional; without it, cloud routing and fallback report a clear error):
   ```bash
   cp .env.example .env   # then set OPENAI_API_KEY
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

Run the tests with `python -m pytest -q`. They use fake engines, so they don't need Ollama or an API key.

## Execution modes

| Mode | Routing | Fallback |
|------|---------|----------|
| Automatic | Rule-based router picks local or cloud | A failed local request is retried once in the cloud |
| Local only | Bypassed, always local | None |
| Cloud only | Bypassed, always cloud | None |

## Routing rules (`hybrid_router/router.py`)

Rules are deterministic and are checked in this order:

1. **Manual mode**: Local-only or Cloud-only overrides everything (`strategy: manual_override`).
2. **Length**: if prompt + context is longer than `LOCAL_MAX_INPUT_CHARS` (default 6000) → cloud.
3. **Complexity indicators** (keyword heuristics) → cloud:
   - advanced proofs (*prove, theorem, by induction*)
   - advanced math (*eigenvalues, differential equation, derive*)
   - complex debugging (*traceback, segfault, race condition, memory leak*, or a debugging request with 20 or more lines of input)
   - advanced optimization (*optimize … performance/query/runtime, time complexity, big-O*)
   - multi-step reasoning
   - advanced technical analysis (*system design, distributed systems, trade-offs, threat model*)
4. **Category**: everything else is classified (summarization, explanation, creative writing, rewriting, coding,
   math, general Q&A) and goes **local**.

Each decision returns `request_id`, `destination`, `reason`, `strategy` and `category`, and the UI shows the reason.

## Response handling (`hybrid_router/response_handler.py`)

- An empty response, or one that only contains a `<think>` block, is marked failed.
- If the prompt explicitly asks for JSON (e.g. "return JSON", "as JSON"), the response must parse as JSON
  (a single ```` ``` ```` code fence around it is allowed). Required fields are checked when the prompt lists them, either
  quoted (`keys "name" and "age"`) or after a colon (`fields: name, age`).
- Fallback happens only in Automatic mode, only when the request was routed to local, and at most once.
  The final result reports `model: "cloud"`, `fallback_used: true` and the reason the local attempt failed.

## Result structure

```json
{
  "request_id": "req_3f9a1c0b2d4e",
  "model": "local",
  "model_name": "qwen3:1.7b",
  "response": "Binary search works by...",
  "status": "success",
  "fallback_used": false,
  "fallback_reason": null,
  "error": null
}
```

Error codes: `invalid_request`, `empty_response`, `invalid_format`, `timeout`, `service_unavailable`,
`model_not_found`, `generation_failed`, `not_configured`, `auth_failed`, `rate_limited`, `network_error`, `internal_error`.

## Layout

```
app.py                         Streamlit UI
hybrid_router/
  config.py                    Settings from env / .env (the API key is never stored here)
  models.py                    Shared dataclasses: Request, RoutingDecision, InferenceResult
  request_processor.py         Validation, normalization, request IDs, context, format detection
  router.py                    Rule-based routing engine
  inference/prompting.py       Shared system prompt + message formatting for both backends
  inference/local_engine.py    Ollama client
  inference/cloud_engine.py    OpenAI client
  response_handler.py          Validation + bounded fallback
  pipeline.py                  Wires the components together; never raises to the UI
tests/                         pytest suite (86 tests)
```

## Configuration

See `.env.example`. The main settings are `LOCAL_MODEL`, `OLLAMA_BASE_URL`, `LOCAL_TIMEOUT_SECONDS`, `LOCAL_MAX_INPUT_CHARS`,
`OPENAI_MODEL`, `CLOUD_TIMEOUT_SECONDS`, `MAX_PROMPT_CHARS` and `MAX_CONTEXT_MESSAGES`.
