from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass(frozen=True)
class Pair:
    id: str
    still_life: Path
    on_model: Path

def discover_pairs(input_dir: Path, limit: int | None = None) -> List[Pair]:
    stills = sorted((input_dir).glob("still-life-*.jpg"))
    pairs: List[Pair] = []
    for s in stills:
        id_ = s.stem.replace("still-life-", "")
        om = input_dir / f"on-model-{id_}.jpg"
        if om.exists():
            pairs.append(Pair(id=id_, still_life=s, on_model=om))
    return pairs[:limit] if limit else pairs
