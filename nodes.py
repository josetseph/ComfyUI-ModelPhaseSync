"""Model Phase Sync nodes — execution barriers, phase unloads, and phase checkpoints."""

from __future__ import annotations

import gc
import hashlib
import os

import comfy.model_management as model_management
import folder_paths
import torch

MAX_SLOTS = 1000
CONDITIONING_EXT = ".conditioning"

# Saved under output/conditionings and optionally input/conditionings (for Load).
_conditioning_output_dir = os.path.join(folder_paths.get_output_directory(), "conditionings")
_conditioning_input_dir = os.path.join(folder_paths.get_input_directory(), "conditionings")
os.makedirs(_conditioning_output_dir, exist_ok=True)
os.makedirs(_conditioning_input_dir, exist_ok=True)

if "conditionings" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["conditionings"] = (
        [_conditioning_output_dir, _conditioning_input_dir],
        {CONDITIONING_EXT},
    )


class AnyType(str):
    """Wildcard type that compares equal to every ComfyUI type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def _tensors_to_cpu(obj):
    """Recursively move tensors in conditioning structures to CPU for safe reload."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tensors_to_cpu(v) for v in obj)
    if hasattr(obj, "addit_embeds") and isinstance(obj.addit_embeds, dict):
        obj.addit_embeds = {k: _tensors_to_cpu(v) for k, v in obj.addit_embeds.items()}
    return obj


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


class SaveConditioning:
    """Save CLIP/text CONDITIONING to disk and pass it through.

    Use this to split a batch into a CLIP-only workflow and a later
    diffusion workflow (Load Conditioning → KSampler), without keeping
    CLIP loaded across the sample phase.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "conditionings/cond"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model_phase_sync"

    def save(self, conditioning, filename_prefix="conditionings/cond"):
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        file = f"{filename}_{counter:05}_{CONDITIONING_EXT}"
        path = os.path.join(full_output_folder, file)

        payload = _tensors_to_cpu(conditioning)
        torch.save({"conditioning": payload, "format_version": 1}, path)
        print(f"[ModelPhaseSync] Saved conditioning → {path}")

        results = [{"filename": file, "subfolder": subfolder, "type": "output"}]
        return {"ui": {"conditionings": results}, "result": (conditioning,)}


class LoadConditioning:
    """Load CONDITIONING previously written by Save Conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("conditionings")
        if not files:
            files = [""]
        return {
            "required": {
                "conditioning_file": (files,),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "model_phase_sync"

    def load(self, conditioning_file):
        if not conditioning_file:
            raise ValueError(
                "No conditioning files found. Run Save Conditioning first "
                f"(writes under output/conditionings/), or copy .conditioning "
                f"files into input/conditionings/."
            )
        path = folder_paths.get_full_path("conditionings", conditioning_file)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Conditioning file not found: {conditioning_file}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")

        if isinstance(data, dict) and "conditioning" in data:
            conditioning = data["conditioning"]
        else:
            # Compatibility with older single-pair torch.save dumps.
            conditioning = data if isinstance(data, list) else [data]

        conditioning = _tensors_to_cpu(conditioning)
        print(f"[ModelPhaseSync] Loaded conditioning ← {path}")
        return (conditioning,)

    @classmethod
    def IS_CHANGED(cls, conditioning_file):
        path = folder_paths.get_full_path("conditionings", conditioning_file)
        if not path or not os.path.exists(path):
            return float("nan")
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning_file):
        if not conditioning_file:
            return True
        path = folder_paths.get_full_path("conditionings", conditioning_file)
        if path is None or not os.path.exists(path):
            return f"Invalid conditioning file: {conditioning_file}"
        return True


NODE_CLASS_MAPPINGS = {
    "SyncBarrierN": SyncBarrierN,
    "PhaseUnload": PhaseUnload,
    "SaveConditioning": SaveConditioning,
    "LoadConditioning": LoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncBarrierN": "Sync Barrier",
    "PhaseUnload": "Phase Unload",
    "SaveConditioning": "Save Conditioning",
    "LoadConditioning": "Load Conditioning",
}
