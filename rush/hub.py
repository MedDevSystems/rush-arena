"""Soldier networks on the Hugging Face Hub (or a local folder): config.json + model.safetensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_REPO = "koskokos/rush-soldier"


def _local_dir(spec: str) -> tuple[Path, str]:
    if spec.startswith("hf:"):
        from huggingface_hub import snapshot_download
        repo, _, rev = spec[3:].partition("@")
        repo = repo or DEFAULT_REPO
        sub = None
        if repo.count("/") > 1:                                    # hf:owner/name/<variant>
            owner, name, sub = repo.split("/", 2)
            repo = f"{owner}/{name}"
        d = Path(snapshot_download(repo, revision=rev or None, allow_patterns=[f"{sub}/*"] if sub else None))
        return (d / sub if sub else d), f"{repo}{'/' + sub if sub else ''}"
    d = Path(spec[4:] if spec.startswith("net:") else spec)
    return d, d.name


def load_payload(d: Path) -> dict:
    if (d / "model.safetensors").exists():
        from safetensors.torch import load_file
        cfg = json.loads((d / "config.json").read_text())
        return {"config": cfg, "state_dict": load_file(str(d / "model.safetensors")), "step": cfg.get("step", 0)}
    ck = sorted(d.glob("ckpt_*_latest.pt"))
    if ck:
        return torch.load(str(ck[0]), map_location="cpu", weights_only=False)
    raise SystemExit(f"{d}: neither config.json + model.safetensors nor ckpt_*_latest.pt")


def random_soldier(device: Any, seed: int = 0) -> Any:
    """An untrained soldier network with the published architecture (soldier_config.json)."""
    from rush.model15 import Policy15, build_model15
    cfg = json.loads((Path(__file__).resolve().parent / "soldier_config.json").read_text())
    torch.manual_seed(seed)
    return Policy15(build_model15(cfg, device), 0, None)


## @io "hf:<repo>[/<variant>][@rev]" or "net:<dir>", device -> (Policy15, display name)
def load_soldier(spec: str, device: Any) -> tuple[Any, str]:
    from rush.model15 import policy_from_payload
    from rush.world15 import OBS_LAYOUT15, STATE_SIZE15
    d, name = _local_dir(spec)
    payload = load_payload(d)
    pol = policy_from_payload(payload, device, layout=OBS_LAYOUT15, state_size=STATE_SIZE15, name=name)
    return pol, payload["config"].get("display_name") or name
