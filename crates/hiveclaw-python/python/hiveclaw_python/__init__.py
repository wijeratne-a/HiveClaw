import math
import os

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
_SCENT_ELEMS = 4096


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


def _validate_slot_scent(scent: mx.array) -> None:
    if scent.dtype != mx.bfloat16:
        raise ValueError(f"scent must be bfloat16, got {scent.dtype}")
    if scent.size != _SCENT_ELEMS:
        raise ValueError(
            f"Phase C scent must have {_SCENT_ELEMS} bf16 elements, got {scent.size}"
        )


class SlabClient(_SlabClientBase):
    def __init__(self):
        super().__init__()
        self._slab_handle = _mlx.SlabHandle(self.surface_id())

    def write_scent(
        self, slot_index: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Write bf16×4096 scent to Phase C slot `slot_index` (stamps last_write_clock)."""
        _validate_slot(slot_index)
        _validate_slot_scent(scent)
        scent_c = mx.contiguous(scent)
        dep = None if depends is None else depends
        return self._slab_handle.write_slot(int(slot_index), scent_c, dep)

    def read_scent(
        self, slot_index: int, shape: list, *, like: mx.array, depends=None
    ) -> mx.array:
        _validate_slot(slot_index)
        dep = None if depends is None else depends
        return self._slab_handle.read_slot(int(slot_index), shape, like, dep)

    def write_scent_at_offset(
        self, byte_offset: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_write(byte_offset, scent)
        scent_c = mx.contiguous(scent)
        dep = None if depends is None else depends
        return self._slab_handle.write(byte_offset, scent_c, dep)

    def read_scent_at_offset(
        self, byte_offset: int, shape: list, *, like: mx.array, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_read(byte_offset, shape)
        dep = None if depends is None else depends
        return self._slab_handle.read(byte_offset, shape, like, dep)

    def claim_task(self, candidates: mx.array, *, depends=None) -> mx.array:
        """Returns scalar int32: claimed slot index, or -1 on failure."""
        if candidates.dtype != mx.int32:
            raise ValueError(f"candidates must be int32, got {candidates.dtype}")
        agent_id = int(os.getpid()) & 0xFFFFFFFF
        dep = None if depends is None else depends
        return self._slab_handle.claim(candidates, agent_id, dep)

    def release_task(self, slot_index: int, *, depends=None) -> None:
        """Clear claim_flag on the CPU (call when finished with a held slot)."""
        _validate_slot(slot_index)
        _ = depends  # ordering: release is synchronous CPU; MLX graph deps not applied here
        self._slab_handle.release_slot(int(slot_index))

    def inhibit(self, slot_index: int, *, depends=None) -> mx.array:
        agent_id = int(os.getpid()) & 0xFFFFFFFF
        dep = None if depends is None else depends
        return self._slab_handle.inhibit(int(slot_index), agent_id, dep)
