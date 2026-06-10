# -*- coding: utf-8 -*-
"""RAM-conditional action mask.

Derives a per-step mask over the 9 discrete actions from the verified
script / UI state bytes (``0xD438`` and ``0xCF07``) that we surface in the
RAM observation as one-hot features.

Action index map (matches ``env.actions`` order in ``gym_env.py``):

    0 = "" (noop)
    1 = "a"
    2 = "b"
    3 = "left"
    4 = "right"
    5 = "up"
    6 = "down"
    7 = "start"
    8 = "select"

Mask semantics (1 = allowed, 0 = blocked):

| Detected state                                  | Allowed actions                  |
|-------------------------------------------------|----------------------------------|
| Dialog box visible (script=1, text_box=1)       | noop, A, B                       |
| Menu / keyboard overlay (script=1, text_box=0)  | noop, A, B, directional          |
| Walking (script=0), default                     | noop, A, B, directional          |
| Walking (script=0), `allow_menus_walking=True`  | all 9 actions                    |

``start`` / ``select`` are always blocked during walking unless the stage
opts in via the ``allow_menus_walking`` config key (mid-curriculum
stages that genuinely need to open menus — switching pokémon, items, etc.
— flip this on; early-game stages keep it off so the policy can't farm
the start/select penalty space).

The mask is derived deterministically from the RAM observation, so it can
be reconstructed at PPO update time from the stored ``ram_states`` tensor
without persisting an extra buffer field. Callers must pass the *last
timestep* of the RAM observation, not the whole sequence — masking is
based on the current frame's state.
"""

import torch

from .gym_env import RAM_FEATURE_INDEX


# Resolved once at import time. If gym_env's feature key set changes, an
# import-time KeyError beats a silent miscompute at inference.
_IDX_SCRIPT_ACTIVE = RAM_FEATURE_INDEX["script_active"]
_IDX_UI_TEXT_BOX = RAM_FEATURE_INDEX["ui_state_text_box"]

# Action-space layout. Hardcoded against ``env.actions`` because the
# semantics are coupled — changing the env order is a breaking change
# anyway.
ACTION_SIZE = 9
NOOP, A, B, LEFT, RIGHT, UP, DOWN, START, SELECT = range(ACTION_SIZE)
_DIRECTIONAL = (LEFT, RIGHT, UP, DOWN)
_AB = (A, B)


def compute_action_mask(ram_last_step, allow_menus_walking=False):
    """Return a ``(B, ACTION_SIZE)`` mask tensor.

    Parameters
    ----------
    ram_last_step : torch.Tensor
        Shape ``(B, ram_dim)`` — the *current frame* slice of the RAM
        observation. If you have the full sequence ``(B, seq_len, ram_dim)``,
        slice ``[..., -1, :]`` before passing.
    allow_menus_walking : bool
        If True, ``start`` and ``select`` are allowed during walking. Use
        for stages where menu-opening is task-relevant (item use, party
        management). Default False.

    Returns
    -------
    mask : torch.Tensor, shape (B, ACTION_SIZE), float
        ``1.0`` where the action is allowed, ``0.0`` where it is masked.
    """
    device = ram_last_step.device
    batch = ram_last_step.shape[0]
    # script_active and ui_text_box are one-hot indicators ∈ {0.0, 1.0};
    # use >= 0.5 for a robust binary read.
    script_active = ram_last_step[:, _IDX_SCRIPT_ACTIVE] >= 0.5
    text_box = ram_last_step[:, _IDX_UI_TEXT_BOX] >= 0.5
    dialog = script_active & text_box  # (B,)

    mask = torch.ones((batch, ACTION_SIZE), device=device, dtype=ram_last_step.dtype)

    # Block directional in dialog.
    for a in _DIRECTIONAL:
        mask[:, a] = torch.where(dialog, torch.zeros_like(mask[:, a]), mask[:, a])

    # Block start/select everywhere unless explicitly allowed during walking.
    # In dialog or menu overlay (script_active=1) start/select are always
    # blocked; in walking they're blocked unless the stage opts in.
    if allow_menus_walking:
        # Allow only when walking (script_active=0); still blocked under
        # any scripted state.
        for a in (START, SELECT):
            mask[:, a] = torch.where(script_active, torch.zeros_like(mask[:, a]), mask[:, a])
    else:
        mask[:, START] = 0.0
        mask[:, SELECT] = 0.0

    return mask


def compute_action_mask_from_byte_state(d438_byte, cf07_byte, allow_menus_walking=False):
    """Compute the same mask directly from raw byte values. Useful for
    test fixtures or any caller that hasn't built the full RAM vector.
    Returns a list[float] of length ACTION_SIZE."""
    script_active = int(d438_byte) == 255
    text_box = int(cf07_byte) == 7
    dialog = script_active and text_box

    mask = [1.0] * ACTION_SIZE
    if dialog:
        for a in _DIRECTIONAL:
            mask[a] = 0.0
    # start / select
    if not allow_menus_walking or script_active:
        mask[START] = 0.0
        mask[SELECT] = 0.0
    return mask
