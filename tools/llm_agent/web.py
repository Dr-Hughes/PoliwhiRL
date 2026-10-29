# -*- coding: utf-8 -*-
"""Web dashboard for the LLM agent: watch, pause, save, stop and restart one
run from a browser on the local network.

    uv run python -m tools.llm_agent.web                  # http://<this host>:8765
    uv run python -m tools.llm_agent.web --port 9000 --model ornith-1.5-9b

The server owns at most one agent process (``python -m tools.llm_agent``),
started in its own session so it outlives a server restart; the server
re-attaches to it through ``Training Outputs/llm_agent/.web_current.json``.
It talks to the agent through two files in the run dir: it writes
``control.json`` ({"paused", "save"}), which the agent polls between turns,
and reads the ``status.json`` the agent rewrites after every turn.

The agent only reads the training config; it never writes to a training run's
output dir (Go-Explore capture and recording are off in game.build_config),
so it is safe to run alongside training.

No authentication: anyone on the network who can reach the port can start and
stop runs. Bind to 127.0.0.1 (--host) if that is not what you want.

An API key typed into the Model server panel is held in the page (and its
sessionStorage) only: it travels to POST /api/validate and, for a run, to the
agent subprocess as LLM_API_KEY — never into argv, run dirs or .web_current.json.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from tools.llm_agent.goals import LADDER  # noqa: E402

RUNS = REPO / "Training Outputs" / "llm_agent"
CURRENT = RUNS / ".web_current.json"
UI = Path(__file__).with_name("web_ui.html")
# The model server's LAN address (192.168.0.15) is unreachable from Homebrew
# Python on macOS (Local Network privacy blocks unsigned binaries; curl is
# exempt), so the default goes through the same host's Tailscale name.
DEFAULT_URL = "http://pingu.tailed5a29.ts.net:11434/v1"
DEFAULT_MODEL = "ornith-1.5-9b"
GOAL_KEYS = [g.key for g in LADDER]


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except OSError, ValueError:
        return default


def _tail_bytes(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            data = f.read().decode("utf-8", "replace")
        return data if size <= n else data.split("\n", 1)[-1]
    except OSError:
        return ""


def _alive(pid: int) -> bool:
    if pid <= 0:  # kill(0, ...) would signal our own process group
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError, TypeError:
        return False


def _inside(path: Path, root: Path = RUNS) -> Path:
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"{path} is outside {root}")
    return path


def _list_models(
    url: str, api_key: str | None = None, timeout: float = 8.0
) -> list[str]:
    """GET {url}/models on an OpenAI-compatible server. The key travels only as
    a Bearer header, never in the URL."""
    req = urllib.request.Request(
        url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key or 'local'}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


class Runner:
    """The one agent process this server controls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.current = _read_json(CURRENT, None)  # {pid, run_dir, cmd, started}

    # -- process state ---------------------------------------------------------
    def running(self) -> bool:
        if self.proc is not None:
            if self.proc.poll() is None:
                return True
            self.current = {**(self.current or {}), "exit_code": self.proc.returncode}
            self.proc = None
            CURRENT.write_text(json.dumps(self.current, indent=2))
            return False
        return bool(
            self.current
            and self.current.get("exit_code") is None
            and _alive(int(self.current.get("pid") or 0))
        )

    @property
    def run_dir(self) -> Path | None:
        return Path(self.current["run_dir"]) if self.current else None

    # -- control file ----------------------------------------------------------
    def control(self, action: str) -> dict:
        if not self.running():
            raise RuntimeError("no run is active")
        assert self.current is not None
        path = Path(self.current["run_dir"]) / "control.json"
        c = _read_json(path, {}) or {}
        if action == "pause":
            c["paused"] = True
        elif action == "resume":
            c["paused"] = False
        elif action == "save":
            c["save"] = int(c.get("save", 0)) + 1
        else:
            raise ValueError(f"unknown action {action!r}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(c))
        tmp.replace(path)
        return c

    # -- start / stop ----------------------------------------------------------
    def start(self, req: dict) -> dict:
        with self.lock:
            if self.running():
                raise RuntimeError("a run is already active: stop it first")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            mode = req.get("mode", "new")
            api_key = str(req.get("api_key") or "").strip()
            cmd = [
                sys.executable,
                "-m",
                "tools.llm_agent",
                "--url",
                str(req.get("url") or DEFAULT_URL),
                "--model",
                str(req.get("model") or DEFAULT_MODEL),
                "--temperature",
                str(float(req.get("temperature", 0.3))),
                "--max-attempts",
                str(int(req.get("max_attempts", 3))),
                "--budget-scale",
                str(float(req.get("budget_scale", 1.0))),
            ]
            if int(req.get("max_steps") or 0) > 0:
                cmd += ["--max-steps", str(int(req["max_steps"]))]
            if req.get("until"):
                cmd += ["--until", str(req["until"])]
            if not req.get("thinking", False):
                cmd.append("--no-thinking")
            if req.get("no_video"):
                cmd.append("--no-video")
            if mode == "new":
                out = RUNS / f"run_{stamp}"
            elif mode == "resume":
                # A run dir (its latest autosave) or one states/autosave_*.json.
                src = _inside(Path(req["save"]))
                base = src.parent.parent if src.is_file() else src
                out = base.with_name(
                    f"{base.name.split('_resumed_')[0]}_resumed_{stamp}"
                )
                cmd += ["--resume", str(src)]
            elif mode == "state":
                state = _inside(Path(req["save"]))
                goal = req.get("goal") or self.goal_for_state(state.name)
                if goal not in GOAL_KEYS:
                    raise ValueError(f"pick a goal to start {state.name} from")
                base = state.parent.parent
                out = base.with_name(
                    f"{base.name.split('_from_')[0]}_from_{goal}_{stamp}"
                )
                cmd += ["--state", str(state), "--from", goal]
            else:
                raise ValueError(f"unknown mode {mode!r}")
            cmd += ["--out", str(out)]
            out.mkdir(parents=True, exist_ok=True)
            (out / "control.json").write_text(json.dumps({"paused": False, "save": 0}))
            log = open(out / "console.log", "ab")
            # The key travels as an env var, never argv: argv would show it in
            # ps and in the cmd record saved to .web_current.json.
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if api_key:
                env["LLM_API_KEY"] = api_key
            # nice: training shares this machine and must not slow down.
            self.proc = subprocess.Popen(
                ["nice", "-n", "5", *cmd],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            log.close()
            self.current = {
                "pid": self.proc.pid,
                "run_dir": str(out),
                "cmd": cmd,
                "mode": mode,
                "started": time.time(),
                "exit_code": None,
            }
            CURRENT.write_text(json.dumps(self.current, indent=2))
            return self.current

    def stop(self) -> dict:
        if not self.running():
            raise RuntimeError("no run is active")
        assert self.current is not None
        pid = self.current["pid"]
        # SIGINT -> KeyboardInterrupt: the agent's finally block writes the
        # report, closes videos and the notebook. An LLM call in flight
        # finishes first; SIGKILL only if it is still up after two minutes.
        os.kill(pid, signal.SIGINT)

        def reap():
            deadline = time.time() + 120
            while time.time() < deadline and self.running():
                time.sleep(1)
            if self.running():
                os.kill(pid, signal.SIGKILL)

        threading.Thread(target=reap, daemon=True).start()
        return {"stopping": pid}

    @staticmethod
    def goal_for_state(name: str) -> str | None:
        """NN_<goal>_start.state -> goal; NN_<goal>_done.state -> the next goal."""
        stem = name.rsplit(".", 1)[0]
        for suffix, offset in (("_start", 0), ("_done", 1)):
            if stem.endswith(suffix) and stem[:2].isdigit():
                key = stem[3 : -len(suffix)]
                if key in GOAL_KEYS:
                    i = GOAL_KEYS.index(key) + offset
                    return GOAL_KEYS[i] if i < len(GOAL_KEYS) else None
        return None

    # -- views -----------------------------------------------------------------
    def saves(self, run_dir: Path) -> list[dict]:
        out = []
        for p in sorted(
            (run_dir / "states").glob("*.state"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            meta = _read_json(p.with_suffix(".json"))
            row = {
                "name": p.name,
                "path": str(p.relative_to(RUNS)),
                "mtime": p.stat().st_mtime,
            }
            if meta:
                row.update(
                    kind="save",
                    goal=meta.get("goal"),
                    goal_steps=meta.get("goal_steps"),
                    json=str(p.with_suffix(".json").relative_to(RUNS)),
                )
            elif p.name.startswith("autosave_"):
                row.update(kind="autosave (state only)", goal=None)
            else:
                row.update(
                    kind="goal start" if p.stem.endswith("_start") else "goal done",
                    goal=self.goal_for_state(p.name),
                )
            out.append(row)
        return out

    def runs(self, limit=25) -> list[dict]:
        rows = []
        for d in RUNS.iterdir() if RUNS.exists() else []:
            if d.is_dir() and (d / "states").is_dir():
                st = _read_json(d / "status.json") or {}
                rows.append(
                    {
                        "name": d.name,
                        "mtime": d.stat().st_mtime,
                        "autosave": (d / "autosave.json").exists(),
                        "reached": st.get("reached")
                        or (_read_json(d / "summary.json") or {}).get("reached"),
                        "report": (d / "REPORT.md").exists(),
                    }
                )
        return sorted(rows, key=lambda r: r["mtime"], reverse=True)[:limit]

    def snapshot(self, view: str | None = None) -> dict:
        running = self.running()
        run_dir = _inside(Path(view)) if view else self.run_dir
        res = {
            "running": running,
            "current": self.current,
            "ladder": [
                {"key": g.key, "title": g.title, "optional": g.optional} for g in LADDER
            ],
            "defaults": {"url": DEFAULT_URL, "model": DEFAULT_MODEL},
            "runs": self.runs(),
        }
        if run_dir and run_dir.exists():
            res["run"] = run_dir.name
            res["is_current"] = bool(self.run_dir and run_dir == self.run_dir)
            res["status"] = _read_json(run_dir / "status.json")
            res["control"] = _read_json(run_dir / "control.json", {})
            res["saves"] = self.saves(run_dir)
            log = _tail_bytes(run_dir / "console.log", 40000)
            # tqdm redraws with \r: keep only the last redraw of each line.
            res["log"] = "\n".join(
                line.rsplit("\r", 1)[-1] for line in log.split("\n")[-80:]
            )
            turns = []
            for line in _tail_bytes(run_dir / "turns.jsonl", 400000).splitlines()[-15:]:
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                d, s = t.get("decision") or {}, t.get("state") or {}
                turns.append(
                    {
                        "turn": t["turn"],
                        "goal": t["goal"],
                        "attempt": t["attempt"],
                        "map": s.get("map"),
                        "x": s.get("x"),
                        "y": s.get("y"),
                        "observation": d.get("observation"),
                        "plan": d.get("plan"),
                        "pressed": t.get("pressed"),
                        "result": t.get("result"),
                        "latency": t.get("latency"),
                        "error": t.get("error"),
                    }
                )
            res["turns"] = turns[::-1]
        return res


RUNNER = Runner()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _file(self, path: Path):
        types = {
            ".png": "image/png",
            ".md": "text/plain; charset=utf-8",
            ".json": "application/json",
            ".mp4": "video/mp4",
            ".log": "text/plain; charset=utf-8",
            ".jsonl": "text/plain; charset=utf-8",
            ".html": "text/html; charset=utf-8",
        }
        try:
            body = path.read_bytes()
        except OSError:
            return self._send(404, b"not found", "text/plain")
        self._send(200, body, types.get(path.suffix, "application/octet-stream"))

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = {k: v[0] for k, v in parse_qs(query).items()}
        path = urllib.parse.unquote(path)
        try:
            if path == "/":
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(RUNNER.snapshot(params.get("run")))
            if path == "/api/models":
                return self._json(_list_models(params.get("url") or DEFAULT_URL))
            if path.startswith("/files/"):
                return self._file(_inside(Path(path[len("/files/") :])))
        except Exception as e:  # noqa: BLE001 - report any failure to the page
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/control":
                return self._json(RUNNER.control(body.get("action", "")))
            if self.path == "/api/stop":
                return self._json(RUNNER.stop())
            if self.path == "/api/start":
                return self._json(RUNNER.start(body))
            if self.path == "/api/validate":
                # Loads the model list from the endpoint (with the key, if given)
                # so the page can show the connection works before starting a run.
                models = _list_models(
                    body.get("url") or DEFAULT_URL, body.get("api_key") or None
                )
                return self._json({"ok": True, "models": models})
        except Exception as e:  # noqa: BLE001 - report any failure to the page
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        self._send(404, b"not found", "text/plain")


def main():
    global DEFAULT_URL, DEFAULT_MODEL
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--url", default=DEFAULT_URL, help="default model server for new runs"
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="default model for new runs")
    args = p.parse_args()
    DEFAULT_URL, DEFAULT_MODEL = args.url, args.model
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"LLM agent dashboard on http://{args.host}:{args.port}  (runs in {RUNS})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
