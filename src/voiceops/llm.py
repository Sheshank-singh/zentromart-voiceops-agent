"""LLM wrapper. Speaks Anthropic-style messages to the rest of the code; the backend is
Google Gemini (default when GEMINI_API_KEY is set) or Anthropic Claude.

Callers pass `messages` as a list of {"role", "content"} where content is a string, a list
of blocks returned by a previous `create()` call, or a list of tool_result dicts. Responses
expose `.content` (blocks with .type "text" / "tool_use"), `.stop_reason` and are read
with `text_of()`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------
# Provider-neutral response shape
# ---------------------------------------------------------------------------------
@dataclass
class Block:
    type: str                      # "text" or "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


class Blocks(list):
    """Assistant content; keeps the provider's raw turn so it can be replayed verbatim
    (Gemini requires its thought signatures back on the next request)."""
    raw = None


@dataclass
class Response:
    content: Blocks
    stop_reason: str               # "end_turn", "tool_use", "max_tokens" or "refusal"


def text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


def create(system: str, messages: list, tools: list | None = None, *, model: str | None = None,
           effort: str | None = None, max_tokens: int = 8000, output_format: dict | None = None):
    backend = {"sarvam": _sarvam_create, "gemini": _gemini_create}.get(config.LLM_PROVIDER, _anthropic_create)
    return backend(system, messages, tools, model=model or config.LLM_MODEL,
                   effort=effort or config.LLM_EFFORT, max_tokens=max_tokens, output_format=output_format)


# ---------------------------------------------------------------------------------
# Sarvam (chat-completions format)
# ---------------------------------------------------------------------------------
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
_sclient = None


def _sarvam():
    global _sclient
    if _sclient is None:
        import httpx
        _sclient = httpx.Client(timeout=120, headers={
            "api-subscription-key": config.SARVAM_API_KEY,
            "Authorization": f"Bearer {config.SARVAM_API_KEY}",
        })
    return _sclient


def _to_chat_messages(system: str, messages: list) -> list:
    out = [{"role": "system", "content": system}]
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": "".join(b.text for b in c if b.type == "text") or None}
            calls = [{"id": b.id, "type": "function",
                      "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)}}
                     for b in c if b.type == "tool_use"]
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif isinstance(c, str):
            out.append({"role": "user", "content": c})
        else:  # tool results
            out += [{"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]} for r in c]
    return out


def _sarvam_create(system, messages, tools, *, model, effort, max_tokens, output_format):
    body = {"model": model, "messages": _to_chat_messages(system, messages), "max_tokens": max_tokens,
            "reasoning_effort": effort if effort in ("low", "medium", "high", "max") else "low"}
    if tools:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]
        body["tool_choice"] = "auto"
    if output_format:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "result", "schema": output_format["schema"]}}
    for attempt in range(5):
        r = _sarvam().post(SARVAM_URL, json=body)
        if r.status_code not in RETRY_STATUS or attempt == 4:
            break
        wait = 2 ** attempt * 2
        log.warning("Sarvam %s; retrying in %ss", r.status_code, wait)
        time.sleep(wait)
    if r.status_code != 200:
        raise RuntimeError(f"Sarvam API {r.status_code}: {r.text[:300]}")

    choice = r.json()["choices"][0]
    msg = choice["message"]
    blocks = Blocks()
    if msg.get("content"):
        blocks.append(Block("text", text=msg["content"]))
    for i, tc in enumerate(msg.get("tool_calls") or []):
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append(Block("tool_use", id=tc.get("id") or f"call_{len(messages)}_{i}",
                            name=tc["function"]["name"], input=args))
    finish = choice.get("finish_reason")
    stop = ("tool_use" if any(b.type == "tool_use" for b in blocks) else
            "max_tokens" if finish == "length" else
            "refusal" if finish == "content_filter" or msg.get("refusal") else "end_turn")
    return Response(blocks, stop)


# ---------------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------------
_gclient = None
RETRY_STATUS = {429, 500, 502, 503, 504}


def _gemini():
    global _gclient
    if _gclient is None:
        from google import genai
        _gclient = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gclient


def _to_gemini_contents(messages: list):
    from google.genai import types
    contents, tool_names = [], {}
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant":
            for b in c:
                if b.type == "tool_use":
                    tool_names[b.id] = b.name
            if getattr(c, "raw", None) is not None:
                contents.append(c.raw)
                continue
            parts = [types.Part(text=b.text) if b.type == "text"
                     else types.Part(function_call=types.FunctionCall(id=b.id, name=b.name, args=b.input))
                     for b in c]
            contents.append(types.Content(role="model", parts=parts))
        elif isinstance(c, str):
            contents.append(types.Content(role="user", parts=[types.Part(text=c)]))
        else:  # tool results
            parts = []
            for r in c:
                try:
                    payload = json.loads(r["content"])
                except (TypeError, json.JSONDecodeError):
                    payload = {"result": r["content"]}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=r["tool_use_id"], name=tool_names.get(r["tool_use_id"], "tool"), response=payload)))
            contents.append(types.Content(role="user", parts=parts))
    return contents


def _gemini_create(system, messages, tools, *, model, effort, max_tokens, output_format):
    from google.genai import errors, types
    cfg = dict(
        system_instruction=system,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=effort),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if tools:
        cfg["tools"] = [types.Tool(function_declarations=[types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
            for t in tools])]
    if output_format:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = output_format["schema"]

    contents = _to_gemini_contents(messages)
    for attempt in range(5):
        try:
            resp = _gemini().models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
            break
        except errors.APIError as exc:
            if exc.code not in RETRY_STATUS or attempt == 4:
                raise
            wait = 2 ** attempt * 2
            log.warning("Gemini %s (%s); retrying in %ss", exc.code, exc.status, wait)
            time.sleep(wait)

    cand = resp.candidates[0] if resp.candidates else None
    blocks = Blocks()
    if cand is None or cand.content is None:
        reason = str(getattr(cand, "finish_reason", "") or getattr(resp.prompt_feedback, "block_reason", ""))
        blocks.raw = None
        return Response(blocks, "refusal" if "SAFETY" in reason or "BLOCK" in reason else "end_turn")
    for i, p in enumerate(cand.content.parts or []):
        if p.function_call:
            fc = p.function_call
            blocks.append(Block("tool_use", id=fc.id or f"call_{len(messages)}_{i}", name=fc.name,
                                input=dict(fc.args or {})))
        elif p.text and not p.thought:
            blocks.append(Block("text", text=p.text))
    # Function-call IDs must match on replay; fill any the API left blank.
    for p, b in zip([p for p in cand.content.parts if p.function_call],
                    [b for b in blocks if b.type == "tool_use"]):
        p.function_call.id = b.id
    blocks.raw = cand.content
    finish = str(cand.finish_reason or "")
    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    elif "MAX_TOKENS" in finish:
        stop = "max_tokens"
    elif "SAFETY" in finish or "PROHIBITED" in finish:
        stop = "refusal"
    else:
        stop = "end_turn"
    return Response(blocks, stop)


# ---------------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------------
_aclient = None
_fallbacks_ok = config.USE_FALLBACKS
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _anthropic():
    global _aclient
    if _aclient is None:
        import anthropic
        _aclient = anthropic.Anthropic(max_retries=3)
    return _aclient


def _to_anthropic_messages(messages: list) -> list:
    out = []
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant" and getattr(c, "raw", None) is not None:
            out.append({"role": "assistant", "content": c.raw})  # includes thinking blocks
        else:
            out.append(m)
    return out


def _anthropic_create(system, messages, tools, *, model, effort, max_tokens, output_format):
    import anthropic
    global _fallbacks_ok
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=_to_anthropic_messages(messages),
        output_config={"effort": effort},
    )
    if tools:
        kwargs["tools"] = tools
    if output_format:
        kwargs["output_config"]["format"] = output_format
    resp = None
    if _fallbacks_ok:
        try:
            # On a safety-classifier refusal, the server retries on a fallback model.
            resp = _anthropic().beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        except anthropic.BadRequestError as exc:
            if "fallback" not in str(exc).lower():
                raise
            log.warning("Server-side fallbacks unavailable for this key; continuing without them.")
            _fallbacks_ok = False
    if resp is None:
        resp = _anthropic().messages.create(**kwargs)
    blocks = Blocks(Block("tool_use", id=b.id, name=b.name, input=dict(b.input)) if b.type == "tool_use"
                    else Block("text", text=b.text)
                    for b in resp.content if b.type in ("text", "tool_use"))
    blocks.raw = resp.content
    return Response(blocks, resp.stop_reason)
