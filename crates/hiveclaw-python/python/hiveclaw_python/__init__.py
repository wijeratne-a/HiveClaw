import math
import os
import platform as _platform
import sys

if sys.platform != "darwin" or _platform.machine().lower() not in (
    "arm64",
    "aarch64",
):
    raise NotImplementedError(
        "HiveClaw Phase 1 is a proof-of-concept tightly integrated with Apple Silicon "
        "(Metal/IOSurface) to demonstrate zero-copy latent sharing at near-zero coordination cost. "
        "Full Linux/x86 CPU fallback and CUDA/vLLM support is planned for Phase 2. "
        "See docs/ARCHITECTURE.md for the roadmap."
    )

import mlx.core as mx
import numpy as np

from ._core import SlabClient as _SlabClientBase

try:
    from . import hiveclaw_mlx_ext as _mlx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "hiveclaw_mlx_ext is missing. From repo root run `make python` "
        "(builds PyO3 + the MLX extension into this package)."
    ) from e

# Batched read/write: dummy rows use -1; C++ casts to 0xFFFFFFFF (sentinel, IOSurface no-op).
SENTINEL_SLOT = -1


def _validate_write(byte_offset: int, scent: mx.array, slab_size: int) -> None:
    if scent.dtype != mx.bfloat16:
        raise ValueError(f"scent must be bfloat16, got {scent.dtype}")
    if byte_offset % 2 != 0:
        raise ValueError(f"byte_offset must be 2-byte aligned, got {byte_offset}")
    end = byte_offset + scent.size * 2
    if end > slab_size:
        raise ValueError(
            f"write exceeds slab: {byte_offset} + {scent.size * 2} > {slab_size}"
        )


def _validate_read(byte_offset: int, shape: list, slab_size: int) -> None:
    if byte_offset % 2 != 0:
        raise ValueError(f"byte_offset must be 2-byte aligned, got {byte_offset}")
    n = math.prod(shape)
    end = byte_offset + n * 2
    if end > slab_size:
        raise ValueError(
            f"read exceeds slab: {byte_offset} + {n * 2} > {slab_size}"
        )


def _validate_slot(slot_index: int, n_slots: int) -> None:
    if slot_index < 0 or slot_index >= n_slots:
        raise ValueError(f"slot_index must be in [0, {n_slots}), got {slot_index}")


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


def _validate_batch_slots_arr(slots: mx.array, n_slots: int) -> int:
    """int32 [B]; real slots unique in [0, N_SLOTS); SENTINEL_SLOT (-1) allowed for dummies."""
    if slots.dtype != mx.int32:
        raise ValueError(f"batch slots must be int32, got {slots.dtype}")
    if len(slots.shape) != 1:
        raise ValueError(f"batch slots must be 1-D [B], got shape {slots.shape}")
    mx.eval(slots)
    flat = np.array(slots, dtype=np.int32).reshape(-1)
    if flat.size == 0:
        raise ValueError("batch slots must be non-empty")
    real = flat[flat != SENTINEL_SLOT]
    if real.size > 0 and np.unique(real).size != real.size:
        raise ValueError("batch slots: duplicate real slot_index")
    for x in real.tolist():
        ix = int(x)
        if ix < 0 or ix >= n_slots:
            raise ValueError(f"slot index out of range: {x}")
    return int(flat.size)


def _validate_batch_latents(latents: mx.array, B: int, d: int) -> None:
    sh = tuple(int(x) for x in latents.shape)
    if sh != (B, 1, d) or latents.dtype != mx.bfloat16:
        raise ValueError(
            f"latents must be [{B},1,{d}] bfloat16, got {sh} {latents.dtype}"
        )


class SlabClient(_SlabClientBase):
    def __init__(self):
        super().__init__()
        self._slab_handle = _mlx.SlabHandle(self.surface_id())

    def get_latent_dim(self) -> int:
        """bf16 latent width per slot (from IOSurface global header v6)."""
        return int(super().get_latent_dim())

    def _slab_size(self) -> int:
        return int(super().slab_size())

    def _n_slots(self) -> int:
        return int(super().n_slots())

    def read_slot_v5(
        self, slot_index: int, *, depends=None
    ) -> mx.array:
        """Read [1,1,D] bf16; torn epoch → zeros (handled in C++)."""
        _validate_slot(slot_index, self._n_slots())
        latent_c = (
            self._slab_handle.read_slot_v5(int(slot_index))
            if depends is None
            else self._slab_handle.read_slot_v5(int(slot_index), depends)
        )
        return latent_c

    def write_slot_v5(
        self, slot_index: int, latent: mx.array, *, depends=None
    ) -> mx.array:
        """Write strict [1,1,D] bfloat16; stamps epochs in C++."""
        _validate_slot(slot_index, self._n_slots())
        sh = tuple(int(x) for x in latent.shape)
        d = self.get_latent_dim()
        if sh != (1, 1, d):
            raise ValueError(
                f"latent must be [1,1,{d}] bfloat16, got {sh} {latent.dtype}"
            )
        if latent.dtype != mx.bfloat16:
            raise ValueError(f"latent must be bfloat16, got {latent.dtype}")
        latent_c = mx.contiguous(latent)
        if depends is None:
            return self._slab_handle.write_slot_v5(int(slot_index), latent_c)
        return self._slab_handle.write_slot_v5(int(slot_index), latent_c, depends)

    def read_slots(
        self, slots: mx.array, *, depends=None
    ) -> tuple[mx.array, mx.array]:
        """Read B slots: returns ([B,1,D] bf16, [B] uint8 status: 0=ok, 1=torn)."""
        _ = _validate_batch_slots_arr(slots, self._n_slots())
        slots_c = mx.contiguous(slots)
        if depends is None:
            return self._slab_handle.read_slots_v5(slots_c)
        return self._slab_handle.read_slots_v5(slots_c, depends)

    def write_slots(
        self, slots: mx.array, latents: mx.array, *, depends=None
    ) -> tuple[mx.array, mx.array]:
        """Write B slots: latents [B,1,D] bf16; returns (latents, [B] uint8 status)."""
        B = _validate_batch_slots_arr(slots, self._n_slots())
        _validate_batch_latents(latents, B, self.get_latent_dim())
        slots_c = mx.contiguous(slots)
        latents_c = mx.contiguous(latents)
        if depends is None:
            return self._slab_handle.write_slots_v5(slots_c, latents_c)
        return self._slab_handle.write_slots_v5(slots_c, latents_c, depends)

    def write_scent(
        self, slot_index: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Write bf16 latent vector to Phase C slot `slot_index` (stamps epochs)."""
        _validate_slot(slot_index, self._n_slots())
        _validate_slot_latent(scent, self.get_latent_dim())
        scent_c = mx.contiguous(scent)
        if depends is None:
            return self._slab_handle.write_slot(int(slot_index), scent_c)
        return self._slab_handle.write_slot(int(slot_index), scent_c, depends)

    def read_scent(
        self, slot_index: int, shape: list, *, like=None, depends=None
    ) -> mx.array:
        _validate_slot(slot_index, self._n_slots())
        st = _shape_list(shape)
        _ = like
        if depends is None:
            return self._slab_handle.read_slot(int(slot_index), st)
        return self._slab_handle.read_slot(int(slot_index), st, depends)

    def write_scent_at_offset(
        self, byte_offset: int, scent: mx.array, *, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_write(byte_offset, scent, self._slab_size())
        scent_c = mx.contiguous(scent)
        if depends is None:
            return self._slab_handle.write(byte_offset, scent_c)
        return self._slab_handle.write(byte_offset, scent_c, depends)

    def read_scent_at_offset(
        self, byte_offset: int, shape: list, *, like=None, depends=None
    ) -> mx.array:
        """Phase B compatibility: raw byte offset into the IOSurface."""
        _validate_read(byte_offset, shape, self._slab_size())
        st = _shape_list(shape)
        _ = like
        if depends is None:
            return self._slab_handle.read(byte_offset, st)
        return self._slab_handle.read(byte_offset, st, depends)

    def read_u32_at(self, byte_offset: int) -> int:
        """Little-endian u32 at byte offset (layout headers; tests / tooling)."""
        bo = int(byte_offset)
        sz = self._slab_size()
        if bo < 0 or bo + 4 > sz:
            raise ValueError(f"read_u32_at out of slab range: {bo}")
        return int(super().read_u32_at(bo))

    def write_u32_at(self, byte_offset: int, value: int) -> None:
        """Little-endian u32 write (e.g. perturb epoch for torn-read tests)."""
        bo = int(byte_offset)
        sz = self._slab_size()
        if bo < 0 or bo + 4 > sz:
            raise ValueError(f"write_u32_at out of slab range: {bo}")
        super().write_u32_at(bo, int(value) & 0xFFFFFFFF)

    def get_slot_states(self) -> list:
        """Best-effort snapshot for all slots."""
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
        _validate_slot(slot_index, self._n_slots())
        _ = depends
        self._slab_handle.release_slot(int(slot_index))

    def inhibit(self, slot_index: int, owner_id: int | None = None, *, depends=None) -> mx.array:
        if owner_id is None:
            owner_id = int(os.getpid()) & 0xFFFFFFFF
        aid = int(owner_id) & 0xFFFFFFFF
        _validate_slot(slot_index, self._n_slots())
        if depends is None:
            return self._slab_handle.inhibit(int(slot_index), aid)
        return self._slab_handle.inhibit(int(slot_index), aid, depends)


from .init import find_repo_root, init, resolve_manager_repo_root
from .local_swarm import AgentConfig, LocalSwarm
from .manager import HiveClawManager
from .swarm import Swarm
