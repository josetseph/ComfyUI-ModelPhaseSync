"""Model Phase Sync nodes — execution barriers and explicit phase unloads."""

from __future__ import annotations

import gc

import comfy.model_management as model_management
import torch

MAX_SLOTS = 1000


class AnyType(str):
    """Wildcard type that compares equal to every ComfyUI type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class SyncBarrierN:
    """True barrier for up to MAX_SLOTS branches.

    An output only becomes available once ALL *connected* inputs are ready.
    Disconnected optional slots do not block. Use this to finish one whole
    phase (e.g. all CLIP encodes, or all KSampler passes) before the next,
    so models are not evicted and reloaded mid-run.

    Sockets auto-grow in the UI (see js/sync_barrier.js); only connect as
    many in_/out_ pairs as you have branches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {f"in_{i}": ("*",) for i in range(1, MAX_SLOTS + 1)},
        }

    RETURN_TYPES = tuple("*" for _ in range(MAX_SLOTS))
    RETURN_NAMES = tuple(f"out_{i}" for i in range(1, MAX_SLOTS + 1))
    FUNCTION = "sync"
    CATEGORY = "model_phase_sync"

    def sync(self, **kwargs):
        return tuple(kwargs.get(f"in_{i}") for i in range(1, MAX_SLOTS + 1))


class PhaseUnload:
    """Passthrough that explicitly frees model memory between workflow phases.

    Drop after a Sync Barrier (or between phases) so the next phase starts
    with a clean slate instead of relying only on ComfyUI's heuristics.
    Optionally pass a specific model to prefer for eviction; otherwise all
    loaded models are unloaded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_type,),
            },
            "optional": {
                "model": (any_type,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "unload"
    CATEGORY = "model_phase_sync"

    def unload(self, value, model=None):
        print("[ModelPhaseSync] Phase Unload:")
        loaded_models = model_management.loaded_models()

        if model is not None and model in loaded_models:
            print(" - Model found in memory, unloading preferentially...")
            loaded_models.remove(model)
            model_management.free_memory(
                1e30, model_management.get_torch_device(), loaded_models
            )
        else:
            if model is not None and isinstance(model, dict) and "model" in model:
                print(f" - Unloading nested model of type {type(model['model']).__name__}")
                del model["model"]
            print(" - Unloading all models...")
            model_management.unload_all_models()

        model_management.soft_empty_cache(True)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            print(" - CUDA cache clear skipped (not available on this device)")

        return (value,)


NODE_CLASS_MAPPINGS = {
    "SyncBarrierN": SyncBarrierN,
    "PhaseUnload": PhaseUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncBarrierN": "Sync Barrier",
    "PhaseUnload": "Phase Unload",
}
