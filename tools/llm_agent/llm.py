# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible chat client (stdlib only) for the omlx server."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from tools.llm_agent.game import BUTTONS

DEFAULT_URL = "https://omlx.tailed5a29.ts.net/v1"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"

# How many buttons one turn may press. A turn costs one model call, so this
# ceiling is the ceiling on progress per call: measured at ~2 s per button
# either way (a faster model that presses fewer buttons is no faster per goal),
# so pressing more per call is the only way to spend less time per goal.
# ``_execute`` stops the list early on anything important, so a long list is
# not a long blind stretch.
MAX_BUTTONS = 10

ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": f"Report what you see, update your plan, and press "
        f"1-{MAX_BUTTONS} buttons.",
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What is literally on screen now: place, people, "
                    "objects, any text box and its words, any menu and where its arrow is.",
                },
                "progress": {
                    "type": "string",
                    "description": "Did your last buttons do what you expected? What changed?",
                },
                "plan": {
                    "type": "string",
                    "description": "Your next 1-3 steps toward the goal.",
                },
                "following": {
                    "type": "string",
                    "description": "Which WALKING ROUTE you are following this turn, written "
                    "exactly as its destination is listed (e.g. CHERRYGROVE_CITY or PROF.ELM), "
                    'or "none". If you name a route, your first button MUST be that '
                    "route's FIRST MOVE.",
                },
                "buttons": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(BUTTONS)},
                    "minItems": 1,
                    "maxItems": MAX_BUTTONS,
                    "description": "Buttons to press in order. Execution stops early if "
                    "something important happens (map change, a scene starts, blocked).",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional. REPLACES your scratchpad: your current working "
                    "plan for this goal. Omit to keep it.",
                },
                "remember": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "description": "Optional. Short facts to ADD to your permanent notebook "
                    "(kept across goals): what a person or sign told you (directions, where "
                    "someone lives, what to do next), what worked, what failed. One fact per "
                    "item, under 30 words. Do not repeat facts already in your notebook.",
                },
                "recall": {
                    "type": "string",
                    "description": "Optional. Keywords to search your notebook with; the "
                    "matching facts are shown to you next turn. Use it when stuck or when "
                    "you need something you learned earlier.",
                },
            },
            "required": ["observation", "progress", "plan", "following", "buttons"],
        },
    },
}


class LLMClient:
    def __init__(
        self,
        url=DEFAULT_URL,
        model=DEFAULT_MODEL,
        thinking=True,
        max_tokens=3072,
        timeout=600,
        api_key=None,
    ):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Local servers (ollama, oMLX) accept any token; hosted OpenAI-compatible
        # endpoints need a real key. Falls back to the LLM_API_KEY env var.
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or "local"

    def _post(self, body):
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def decide(
        self,
        system: str,
        user: str,
        images_b64: list[str],
        temperature: float,
        model: str | None = None,
        thinking: bool | None = None,
    ) -> dict:
        """One decision. Returns {decision, raw, reasoning, usage, latency,
        error, model}."""
        model = model or self.model
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
            for b in images_b64
        ]
        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "tools": [ACT_TOOL],
            "tool_choice": "required",
            "chat_template_kwargs": {
                "enable_thinking": self.thinking if thinking is None else thinking
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        t0 = time.time()
        last_err = None
        retries = []  # every failed try, kept for the logs
        for attempt in range(4):
            try:
                resp = self._post(body)
                if resp.get("choices"):
                    break
                # The server answers transient failures (e.g. a model reload)
                # with an error body instead of choices.
                last_err = f"no choices: {json.dumps(resp)[:300]}"
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                json.JSONDecodeError,
            ) as e:
                last_err = repr(e)
            retries.append(
                {
                    "try": attempt + 1,
                    "after_s": round(time.time() - t0, 1),
                    "error": last_err,
                }
            )
            time.sleep(5 * (attempt + 1))
        else:
            return {
                "decision": None,
                "raw": None,
                "reasoning": None,
                "usage": None,
                "latency": time.time() - t0,
                "error": last_err,
                "model": model,
                "retries": retries,
            }

        msg = resp["choices"][0]["message"]
        raw = None
        if msg.get("tool_calls"):
            raw = msg["tool_calls"][0]["function"]["arguments"]
        elif msg.get("content"):
            raw = msg["content"]
        decision, error = parse_decision(raw)
        reasoning_tokens = (
            (resp.get("usage") or {}).get("completion_tokens_details") or {}
        ).get("reasoning_tokens", 0)
        if (
            decision is None
            and body["chat_template_kwargs"]["enable_thinking"]
            and (reasoning_tokens or 0) >= self.max_tokens * 0.9
        ):
            # The model reasoned through its whole budget without answering
            # (seen on the naming keyboard and long battles): ask again
            # without thinking rather than lose the turn.
            retry = self.decide(
                system, user, images_b64, temperature, model=model, thinking=False
            )
            retry["latency"] += time.time() - t0
            retry["error"] = retry["error"] or None
            retry["retried_without_thinking"] = True
            return retry
        return {
            "decision": decision,
            "raw": raw,
            "content": msg.get("content"),
            "reasoning": msg.get("reasoning_content"),
            "usage": resp.get("usage"),
            "latency": time.time() - t0,
            "error": error,
            "model": model,
            "retries": retries,
            "finish_reason": resp["choices"][0].get("finish_reason"),
        }


def parse_decision(raw):
    """Tool-call arguments (or a JSON object embedded in plain content) ->
    validated decision dict, or (None, reason)."""
    if not raw:
        return None, "empty response"
    data = None
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
    if not isinstance(data, dict):
        return None, "unparseable response"
    buttons = data.get("buttons")
    if isinstance(buttons, str):
        buttons = re.findall(r"[a-z]+", buttons.lower())
    buttons = [str(b).strip().lower() for b in (buttons or [])]
    buttons = [b for b in buttons if b in BUTTONS][:MAX_BUTTONS]
    if not buttons:
        return None, f"no valid buttons in {data.get('buttons')!r}"
    data["buttons"] = buttons
    remember = data.get("remember")
    if isinstance(remember, str):
        remember = [remember]
    data["remember"] = [str(r).strip() for r in (remember or []) if str(r).strip()][:3]
    recall = data.get("recall")
    data["recall"] = str(recall).strip() if recall else ""
    following = data.get("following")
    data["following"] = str(following).strip()[:60] if following else ""
    return data, None
