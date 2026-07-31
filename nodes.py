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
LATENT_EXT = ".latent"

# Saved under output/conditionings and optionally input/conditionings (for Load).
_conditioning_output_dir = os.path.join(folder_paths.get_output_directory(), "conditionings")
_conditioning_input_dir = os.path.join(folder_paths.get_input_directory(), "conditionings")
os.makedirs(_conditioning_output_dir, exist_ok=True)
os.makedirs(_conditioning_input_dir, exist_ok=True)

_latent_output_dir = os.path.join(folder_paths.get_output_directory(), "latents")
_latent_input_dir = os.path.join(folder_paths.get_input_directory(), "latents")
os.makedirs(_latent_output_dir, exist_ok=True)
os.makedirs(_latent_input_dir, exist_ok=True)

if "conditionings" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["conditionings"] = (
        [_conditioning_output_dir, _conditioning_input_dir],
        {CONDITIONING_EXT},
    )

if "mps_latents" not in folder_paths.folder_names_and_paths:
    # Separate key so we don't collide with any future core "latents" registration.
    folder_paths.folder_names_and_paths["mps_latents"] = (
        [
            _latent_output_dir,
            _latent_input_dir,
            folder_paths.get_input_directory(),
        ],
        {LATENT_EXT},
    )


class AnyType(str):
    """Wildcard type that compares equal to every ComfyUI type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

_NO_CONDITIONING_FILES = (
    "no .conditioning files found — run Save Conditioning first "
    "(or put files in input/conditionings or output/conditionings)"
)

_NO_LATENT_FILES = (
    "no .latent files found — run Save Latent first "
    "(or upload / put files in input/latents or output/latents)"
)


def _invalidate_conditioning_file_cache() -> None:
    folder_paths.filename_list_cache.pop("conditionings", None)


def _invalidate_latent_file_cache() -> None:
    folder_paths.filename_list_cache.pop("mps_latents", None)


def _list_conditioning_files() -> list[str]:
    """Fresh file list (bypass stale folder_paths cache)."""
    _invalidate_conditioning_file_cache()
    try:
        return folder_paths.get_filename_list("conditionings")
    except Exception:
        files: list[str] = []
        for root in (_conditioning_output_dir, _conditioning_input_dir):
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    if name.endswith(CONDITIONING_EXT):
                        rel = os.path.relpath(os.path.join(dirpath, name), root)
                        files.append(rel.replace("\\", "/"))
        return sorted(set(files))


def _list_latent_files() -> list[str]:
    _invalidate_latent_file_cache()
    try:
        return folder_paths.get_filename_list("mps_latents")
    except Exception:
        files = []
        roots = (
            _latent_output_dir,
            _latent_input_dir,
            folder_paths.get_input_directory(),
        )
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    if name.endswith(LATENT_EXT):
                        rel = os.path.relpath(os.path.join(dirpath, name), root)
                        files.append(rel.replace("\\", "/"))
        return sorted(set(files))


def _resolve_under_bases(name: str, bases: list[str]) -> str | None:
    if os.path.isfile(name):
        return name
    for base in bases:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_conditioning_path(conditioning_file: str = "", path: str = "") -> str:
    """Resolve dropdown selection or a typed/relative/absolute path override."""
    path = (path or "").strip()
    if path:
        resolved = _resolve_under_bases(
            path,
            [
                folder_paths.get_output_directory(),
                folder_paths.get_input_directory(),
                _conditioning_output_dir,
                _conditioning_input_dir,
            ],
        )
        if resolved:
            return resolved
        raise FileNotFoundError(f"Conditioning path not found: {path}")

    conditioning_file = (conditioning_file or "").strip()
    if (
        not conditioning_file
        or conditioning_file.startswith("no .conditioning files")
    ):
        raise ValueError(
            "No conditioning file selected. Run Save Conditioning first, then "
            "refresh this node (or re-add it) so the dropdown updates — upload via "
            "the button — or paste a path into the path field."
        )

    resolved = folder_paths.get_full_path("conditionings", conditioning_file)
    if resolved and os.path.isfile(resolved):
        return resolved
    resolved = _resolve_under_bases(
        conditioning_file,
        [_conditioning_output_dir, _conditioning_input_dir],
    )
    if resolved:
        return resolved
    raise FileNotFoundError(f"Conditioning file not found: {conditioning_file}")


def _resolve_latent_path(latent_file: str = "", path: str = "") -> str:
    path = (path or "").strip()
    if path:
        # Support Load Latent-style annotated names and input/output relatives.
        if folder_paths.exists_annotated_filepath(path):
            return folder_paths.get_annotated_filepath(path)
        resolved = _resolve_under_bases(
            path,
            [
                folder_paths.get_output_directory(),
                folder_paths.get_input_directory(),
                _latent_output_dir,
                _latent_input_dir,
            ],
        )
        if resolved:
            return resolved
        raise FileNotFoundError(f"Latent path not found: {path}")

    latent_file = (latent_file or "").strip()
    if not latent_file or latent_file.startswith("no .latent files"):
        raise ValueError(
            "No latent file selected. Run Save Latent / upload a .latent file, "
            "or paste a path into the path field."
        )

    if folder_paths.exists_annotated_filepath(latent_file):
        return folder_paths.get_annotated_filepath(latent_file)

    resolved = folder_paths.get_full_path("mps_latents", latent_file)
    if resolved and os.path.isfile(resolved):
        return resolved
    resolved = _resolve_under_bases(
        latent_file,
        [
            _latent_output_dir,
            _latent_input_dir,
            folder_paths.get_input_directory(),
            folder_paths.get_output_directory(),
        ],
    )
    if resolved:
        return resolved
    raise FileNotFoundError(f"Latent file not found: {latent_file}")


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
        _invalidate_conditioning_file_cache()
        print(f"[ModelPhaseSync] Saved conditioning → {path}")

        results = [{"filename": file, "subfolder": subfolder, "type": "output"}]
        return {"ui": {"conditionings": results}, "result": (conditioning,)}


class LoadConditioning:
    """Load CONDITIONING previously written by Save Conditioning.

    Includes a JS "choose file to upload" button (Load Image's built-in
    image_upload flag only works for images). Uploads land in
    input/conditionings/. You can also pick from the dropdown of files in
    output/conditionings and input/conditionings, or paste a path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_conditioning_files()
        return {
            "required": {
                "conditioning_file": (files if files else [_NO_CONDITIONING_FILES],),
            },
            "optional": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional path override. Absolute, or relative to "
                            "output/ / input/ / conditionings/ "
                            "(e.g. conditionings/cond_00001_.conditioning)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "model_phase_sync"

    def load(self, conditioning_file, path=""):
        resolved = _resolve_conditioning_path(conditioning_file, path)

        try:
            data = torch.load(resolved, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(resolved, map_location="cpu")

        if isinstance(data, dict) and "conditioning" in data:
            conditioning = data["conditioning"]
        else:
            # Compatibility with older single-pair torch.save dumps.
            conditioning = data if isinstance(data, list) else [data]

        conditioning = _tensors_to_cpu(conditioning)
        print(f"[ModelPhaseSync] Loaded conditioning ← {resolved}")
        return (conditioning,)

    @classmethod
    def IS_CHANGED(cls, conditioning_file, path=""):
        try:
            resolved = _resolve_conditioning_path(conditioning_file, path)
        except Exception:
            return float("nan")
        m = hashlib.sha256()
        with open(resolved, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning_file, path=""):
        if (path or "").strip():
            return True
        if not conditioning_file or conditioning_file.startswith("no .conditioning"):
            return True
        try:
            _resolve_conditioning_path(conditioning_file, path)
        except Exception as exc:
            return str(exc)
        return True


class LoadLatentUpload:
    """Load a .latent file with an upload button (built-in Load Latent has none).

    Compatible with ComfyUI Save Latent files. Upload goes to input/latents/.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_latent_files()
        return {
            "required": {
                "latent_file": (files if files else [_NO_LATENT_FILES],),
            },
            "optional": {
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional path override. Absolute, or relative to "
                            "output/ / input/ / latents/ "
                            "(e.g. latents/zimage_01_00001_.latent)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "load"
    CATEGORY = "model_phase_sync"

    def load(self, latent_file, path=""):
        import safetensors.torch

        resolved = _resolve_latent_path(latent_file, path)
        latent = safetensors.torch.load_file(resolved, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        print(f"[ModelPhaseSync] Loaded latent ← {resolved}")
        return (samples,)

    @classmethod
    def IS_CHANGED(cls, latent_file, path=""):
        try:
            resolved = _resolve_latent_path(latent_file, path)
        except Exception:
            return float("nan")
        m = hashlib.sha256()
        with open(resolved, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, latent_file, path=""):
        if (path or "").strip():
            return True
        if not latent_file or latent_file.startswith("no .latent"):
            return True
        try:
            _resolve_latent_path(latent_file, path)
        except Exception as exc:
            return str(exc)
        return True


NODE_CLASS_MAPPINGS = {
    "SyncBarrierN": SyncBarrierN,
    "PhaseUnload": PhaseUnload,
    "SaveConditioning": SaveConditioning,
    "LoadConditioning": LoadConditioning,
    "LoadLatentUpload": LoadLatentUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncBarrierN": "Sync Barrier",
    "PhaseUnload": "Phase Unload",
    "SaveConditioning": "Save Conditioning",
    "LoadConditioning": "Load Conditioning",
    "LoadLatentUpload": "Load Latent (Upload)",
}
