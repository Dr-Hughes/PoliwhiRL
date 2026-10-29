# RAM Mapping — moved

The RAM address and event-flag reference now lives in [`docs/reference/ram-mapping.md`](docs/reference/ram-mapping.md) — the canonical copy, verified against live emulator reads.

This is a pointer stub: it forwards to the canonical page in **one hop**, so an old bookmark on this path still lands somewhere useful. It is not the reference itself, and it will not be updated — always open `docs/reference/ram-mapping.md`.

> **Known gap.** In-code comments are *not* redirected here. Eight docstring/comment pointers still name this old root path and therefore resolve to this stub rather than to the canonical page: `PoliwhiRL/environment/gym_env.py:138`, `:355`, `:1114` and `PoliwhiRL/environment/RAM.py:83`, `:88`, `:124`, `:346`. They are pending a separate code-comment cleanup pass, tracked in [`docs/roadmap.md`](docs/roadmap.md) under **Open work → Docs**.
