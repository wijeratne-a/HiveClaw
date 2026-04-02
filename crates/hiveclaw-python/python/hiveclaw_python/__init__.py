import json
import math
import os
import sys
import time

import mlx.core as mx

from ._core import SlabClient as _SlabClientBase

try:
    from . import hiveclaw_mlx_ext as _mlx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "hiveclaw_mlx_ext is missing. From repo root run `make python` "
        "(builds PyO3 + the MLX extension into this package)."
    ) from e

_SLAB_SIZE = 4_718_720
_N_SLOTS = 32
# Slab v4 Phase-C (must match hiveclaw-core math.rs)
_PHASE_C_GLOBAL_HDR = 128
_SLOT_STRIDE_V4 = 4224
_OFF_S_FRONT_EPOCH = 12
_OFF_SLOT_BACK_EPOCH = 4160


def _telemetry_log(obj: dict) -> None:
    o = dict(obj)
    o.setdefault("ts_ns", time.time_ns())
    sys.stderr.write(json.dumps(o, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def _slot_byte_base(slot_index: int) -> int:
    return _PHASE_C_GLOBAL_HDR + int(slot_index) * _SLOT_STRIDE_V4


def _validate_write(byte_offset: int, scent: mx.array) -> None:
    if scent.dtype != mx.bfloat16:
        raise ValueError(f"scent must be bfloat16, got {scent.dtype}")
    if byte_offset % 2 != 0:
        raise ValueError(f"byte_offset must be 2-byte aligned, got {byte_offset}")
    end = byte_offset + scent.size * 2
    if end > _SLAB_SIZE:
        raise ValueError(
            f"write exceeds slab: {byte_offset} + {scent.size * 2} > {_SLAB_SIZE}"
        )


def _validate_read(byte_offset: int, shape: list) -> None:
    if byte_offset % 2 != 0:
        raise ValueError(f"byte_offset must be 2-byte aligned, got {byte_offset}")
    n = math.prod(shape)
    end = byte_offset + n * 2
    if end > _SLAB_SIZE:
        raise ValueError(
            f"read exceeds slab: {byte_offset} + {n * 2} > {_SLAB_SIZE}"
        )


def _validate_slot(slot_index: int) -> None:
    if slot_index < 0 or slot_index >= _N_SLOTS:
        raise ValueError(f"slot_index must be in [0, {_N_SLOTS}), got {slot_index}")


def _validate_slot_scent(scent: mx.array, expected: int) -> None:
    if scent.dtype != mx.bfloat16:
        raise ValueError(f"scent must be bfloat16, got {scent.dtype}")
    if scent.size != expected:
        raise ValueError(
            f"Phase C scent must have {expected} bf16 elements, got {scent.size}"
        )


def _shape_list(shape: list) -> list[int]:
    """list[int] for nanobind → std::vector<int> (tuple is not always accepted)."""
    return [int(d) for d in shape]


class SlabClient(_SlabClientBase):
    def __init__(self):
        super().__init__()
        self._slab_handle = _mlx.SlabHandle(self.surface_id())

    def get_scent_dim(self) -> int:
        """Number of bf16 elements per Phase C scent slot (compiled `SCENT_ELEMS` in math.rs)."""
        return int(super().get_scent_dim())

    def read_scent_if_consistent(
        self,
        slot_index: int,
        shape: list,
        *,
        like=None,
        depends=None,
        context: str = "read",
    ):
        """Read scent only if front_epoch == back_epoch before and after the read; else None."""
        _validate_slot(slot_index)
        st = _shape_list(shape)
        sb = _slot_byte_base(slot_index)
        fe = self.read_u32_at(sb + _OFF_S_FRONT_EPOCH)
        be = self.read_u32_at(sb + _OFF_SLOT_BACK_EPOCH)
        if fe != be:
            _telemetry_log(
                {
                    "event": "torn_epoch_skip",
                    "context": context,
                    "slot_id": int(slot_index),
                    "front_epoch": fe,
                    "back_epoch": be,
                    "stage": "pre",
                }
            )
            return None
        arr = self.read_scent(slot_index, st, like=like, depends=depends)
        fe2 = self.read_u32_at(sb + _OFF_S_FRONT_EPOCH)
        be2 = self.read_u32_at(sb + _OFF_SLOT_BACK_EPOCH)
        if fe2 != fe or be2 != be or fe2 != be2:
            _telemetry_log(
                {
                    "event": "torn_epoch_skip",
                    "context": context,
                    "slot_id": int(slot_index),
                    "front_epoch": fe2,
                    "back_epoch": be2,
                    "stage": "post",
                }
            )
            return None
        return arr

    def read_scent_for_steering(self, slot_index: int, h_step: mx.array, *, depends=None):
        """Epoch-checked slab read for active steering; torn read => zero tensor (silent swarm)."""
        _validate_slot(slot_index)
        sb = _slot_byte_base(slot_index)
        fe = self.read_u32_at(sb + _OFF_S_FRONT_EPOCH)
        be = self.read_u32_at(sb + _OFF_SLOT_BACK_EPOCH)
        if fe != be:
            _telemetry_log(
                {
                    "event": "torn_read_steering_zero",
                    "slot_id": int(slot_index),
                    "front_epoch": fe,
                    "back_epoch": be,
                    "stage": "pre",
                }
            )
            return mx.zeros_like(h_step)
        dep = depends if depends is not None else h_step
        scent = self.read_scent(
            slot_index,
            [1, 1, int(h_step.shape[-1])],
            depends=dep,
        )
        fe2 = self.read_u32_at(sb + _OFF_S_FRONT_EPOCH)
        be2 = self.read_u32_at(sb + _OFF_SLOT_BACK_EPOCH)
        if fe2 != fe or be2 != be or fe2 != be2:
            _telemetry_log(
                {
                    "event": "torn_read_steering_zero",
                    "slot_id": int(slot_index),
                    "front_epoch": fe2,
                    "back_epoch": be2,
                    "stage": "post",
                }
            )
            return mx.zeros_like(h_step)
        return scent

    def write_scent(
        self, slot_index: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Write bf16 scent vector to Phase C slot `slot_index` (stamps last_write_clock)."""
        _validate_slot(slot_index)
        _validate_slot_scent(scent, self.get_scent_dim())
        scent_c = mx.contiguous(scent)
        if depends is None:
            return self._slab_handle.write_slot(int(slot_index), scent_c)
        return self._slab_handle.write_slot(int(slot_index), scent_c, depends)

    def read_scent(
        self, slot_index: int, shape: list, *, like=None, depends=None
    ) -> mx.array:
        _validate_slot(slot_index)
        st = _shape_list(shape)
        # `like` is kept for call-site compatibility; slab reads use the default GPU
        # stream unless `depends` is set (stream is taken from `depends`).
        _ = like
        if depends is None:
            return self._slab_handle.read_slot(int(slot_index), st)
        return self._slab_handle.read_slot(int(slot_index), st, depends)

    def write_scent_at_offset(
        self, byte_offset: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_write(byte_offset, scent)
        scent_c = mx.contiguous(scent)
        if depends is None:
            return self._slab_handle.write(byte_offset, scent_c)
        return self._slab_handle.write(byte_offset, scent_c, depends)

    def read_scent_at_offset(
        self, byte_offset: int, shape: list, *, like=None, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_read(byte_offset, shape)
        st = _shape_list(shape)
        _ = like
        if depends is None:
            return self._slab_handle.read(byte_offset, st)
        return self._slab_handle.read(byte_offset, st, depends)

    def get_slot_states(self) -> list:
        """Best-effort snapshot: [{'claimed': bool, 'owner_id': int}, ...] for 32 slots."""
        raw = self._slab_handle.get_slot_states()
        return [{"claimed": bool(c), "owner_id": int(oid)} for c, oid in raw]

    def claim_task(self, candidates: mx.array, *, depends=None) -> mx.array:
        """Returns scalar int32: claimed slot index, or -1 on failure."""
        if candidates.dtype != mx.int32:
            raise ValueError(f"candidates must be int32, got {candidates.dtype}")
        agent_id = int(os.getpid()) & 0xFFFFFFFF
        if depends is None:
            return self._slab_handle.claim(candidates, agent_id)
        return self._slab_handle.claim(candidates, agent_id, depends)

    def release_task(self, slot_index: int, *, depends=None) -> None:
        """Clear claim_flag on the CPU (call when finished with a held slot)."""
        _validate_slot(slot_index)
        _ = depends  # ordering: release is synchronous CPU; MLX graph deps not applied here
        self._slab_handle.release_slot(int(slot_index))

    def inhibit(self, slot_index: int, owner_id: int | None = None, *, depends=None) -> mx.array:
        if owner_id is None:
            owner_id = int(os.getpid()) & 0xFFFFFFFF
        aid = int(owner_id) & 0xFFFFFFFF
        if depends is None:
            return self._slab_handle.inhibit(int(slot_index), aid)
        return self._slab_handle.inhibit(int(slot_index), aid, depends)
