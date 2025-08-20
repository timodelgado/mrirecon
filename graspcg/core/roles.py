from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence, Union, List, Dict

AxesSpec = Union[str, Sequence[int], Tuple[int, ...]]

@dataclass(frozen=True)
class Roles:
    """
    Describes the semantic partition of axes for an 'image-like' tensor x:

        x.shape == (U..., L..., N...)
                     ^     ^     ^
                   unlike like  nufft
    """
    unlike: int
    like: int
    nufft: int

    def total(self) -> int:
        return int(self.unlike) + int(self.like) + int(self.nufft)

    def token_map(self) -> Dict[str, Tuple[int, ...]]:
        u, l, n = int(self.unlike), int(self.like), int(self.nufft)
        return {
            "temporal": tuple(range(0, u)),
            "time":     tuple(range(0, u)),
            "unlike":   tuple(range(0, u)),
            "like":     tuple(range(u, u + l)),
            "spatial":  tuple(range(u + l, u + l + n)),
            "image":    tuple(range(u + l, u + l + n)),
            "nufft":    tuple(range(u + l, u + l + n)),
            "all":      tuple(range(0, u + l + n)),
            "none":     tuple(),
        }

    def resolve_axes(self, spec: AxesSpec) -> Tuple[int, ...]:
        tm = self.token_map()
        t = self.total()
        if isinstance(spec, str):
            key = spec.lower()
            if key not in tm:
                raise KeyError(f"Unknown axes token '{spec}'.")
            return tm[key]
        if not isinstance(spec, (list, tuple)):
            raise TypeError(f"Bad axes spec type: {type(spec)}")
        out: List[int] = []
        for s in spec:
            if isinstance(s, str):
                key = s.lower()
                if key not in tm:
                    raise KeyError(f"Unknown axes token '{s}'.")
                out.extend(tm[key])
            elif isinstance(s, int):
                ax = s if s >= 0 else (t + s)
                if ax < 0 or ax >= t:
                    raise IndexError(f"Axis {s} out of range for total dims {t}")
                out.append(ax)
            else:
                raise TypeError(f"Bad axes element type: {type(s)}")
        seen = set(); uniq: List[int] = []
        for a in out:
            if a not in seen:
                seen.add(a); uniq.append(a)
        return tuple(uniq)

    @property
    def temporal(self) -> Tuple[int, ...]:
        return self.token_map()["temporal"]

    @property
    def like_axes(self) -> Tuple[int, ...]:
        return self.token_map()["like"]

    @property
    def spatial(self) -> Tuple[int, ...]:
        return self.token_map()["spatial"]
    
    @property
    def ndim(self) -> int:
        return self.total()