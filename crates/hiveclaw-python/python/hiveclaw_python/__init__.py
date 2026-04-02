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
_N_SLOTS = 4096
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


def _validate_slot_latent(latent: mx.array, expected: int) -> None:
    if latent.dtype != mx.bfloat16:
        raise ValueError(f"latent must be bfloat16, got {latent.dtype}")
    if latent.size != expected:
        raise ValueError(
            f"Phase C latent must have {expected} bf16 elements, got {latent.size}"
        )


def _shape_list(shape: list) -> list[int]:
    """list[int] for nanobind → std::vector<int> (tuple is not always accepted)."""
    return [int(d) for d in shape]


class SlabClient(_SlabClientBase):
    def __init__(self):
        super().__init__()
        self._slab_handle = _mlx.SlabHandle(self.surface_id())

    def get_latent_dim(self) -> int:
        """SAE latent width (256); matches `SCENT_ELEMS` / `get_latent_dim()` in MLX ext."""
        return int(_mlx.get_latent_dim())

    def read_slot_v5(
        self, slot_index: int, *, depends=None
    ) -> mx.array:
        """Read [1,1,256] bf16; torn epoch → zeros (handled in C++)."""
        _validate_slot(slot_index)
        latent_c = (
            self._slab_handle.read_slot_v5(int(slot_index))
            if depends is None
            else self._slab_handle.read_slot_v5(int(slot_index), depends)
        )
        return latent_c

    def write_slot_v5(
        self, slot_index: int, latent: mx.array, *, depends=None
    ) -> mx.array:
        """Write strict [1,1,256] bfloat16; stamps epochs in C++."""
        _validate_slot(slot_index)
        sh = tuple(int(x) for x in latent.shape)
        if sh != (1, 1, self.get_latent_dim()):
            raise ValueError(
                f"latent must be [1,1,{self.get_latent_dim()}] bfloat16, got {sh} {latent.dtype}"
            )
        if latent.dtype != mx.bfloat16:
            raise ValueError(f"latent must be bfloat16, got {latent.dtype}")
        latent_c = mx.contiguous(latent)
        if depends is None:
            return self._slab_handle.write_slot_v5(int(slot_index), latent_c)
        return self._slab_handle.write_slot_v5(int(slot_index), latent_c, depends)

    def write_scent(
        self, slot_index: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Write bf16 latent vector to Phase C slot `slot_index` (stamps epochs)."""
        _validate_slot(slot_index)
        _validate_slot_latent(scent, self.get_latent_dim())
        scent_c = mx.contiguous(scent)
        if depends is None:
            return self._slab_handle.write_slot(int(slot_index), scent_c)
        return self._slab_handle.write_slot(int(slot_index), scent_c, depends)

    def read_scent(
        self, slot_index: int, shape: list, *, like=None, depends=None
    ) -> mx.array:
        _validate_slot(slot_index)
        st = _shape_list(shape)
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
        """Best-effort snapshot for all slots (4096 for v5)."""
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
        _ = depends
        self._slab_handle.release_slot(int(slot_index))

    def inhibit(self, slot_index: int, owner_id: int | None = None, *, depends=None) -> mx.array:
        if owner_id is None:
            owner_id = int(os.getpid()) & 0xFFFFFFFF
        aid = int(owner_id) & 0xFFFFFFFF
        if depends is None:
            return self._slab_handle.inhibit(int(slot_index), aid)
        return self._slab_handle.inhibit(int(slot_index), aid, depends)
