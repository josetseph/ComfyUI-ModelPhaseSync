# Model Phase Sync

If your ComfyUI log shows `Unloaded partially` repeating between every prompt in a batched multi-image workflow, this fixes it.

**Stop ComfyUI from reloading CLIP / UNet / VAE between branches** when you batch multiple prompts in one workflow. Force true phase barriers so each model loads once, does all of its work, then yields to the next phase.

## The problem

Batched multi-prompt graphs (several CLIP encodes → several samplers → several VAE decodes) often thrash on constrained VRAM / unified memory (especially Apple Silicon): CLIP loads, gets partially unloaded for UNet, reloads, unloads again… minutes of waste per run.

## The fix

| Node | What it does |
|------|----------------|
| **Sync Barrier** | True execution barrier. Outputs only resolve once all *connected* inputs are ready, so Comfy finishes an entire phase before the next starts. Sockets auto-grow as you wire branches (up to 1000). |
| **Phase Unload** | Optional companion. Passthrough that explicitly evicts models between phases so you are not fully dependent on ComfyUI’s heuristics. |
| **Save / Load Conditioning** | Checkpoint CLIP `CONDITIONING` to disk (and reload it) so you can run text encode as its own workflow, then sample later without CLIP in memory. |
| **Load Latent (Upload)** | Same idea for `.latent` files — built-in Load Latent has no upload button; this one does. Compatible with ComfyUI **Save Latent**. |

### Before / after (3-image batch, Z-Image Turbo on MPS)

| Approach | Wall time | Behavior |
|----------|-----------|----------|
| 3 separate single-image runs | ~1500s | One clean load cycle each, no batching |
| Batched, no barrier | ~1286s | Repeated `Unloaded partially` / reload thrash |
| **Batched + Sync Barrier** | **887s (14:47)** | One CLIP load, one UNet load, one VAE load |

## Install

### ComfyUI-Manager / Registry

Search for **Model Phase Sync** (or `model-phase-sync`) on the [Comfy Registry](https://registry.comfy.org) / Manager:

```bash
comfy node install model-phase-sync
```

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/josetseph/ComfyUI-ModelPhaseSync.git
```

Restart ComfyUI. Nodes appear under **model_phase_sync**.

## How to use

1. Duplicate your per-prompt branch (CLIP encode, sampler, etc.) for each prompt.
2. After each phase’s branch outputs (e.g. all CLIP `CONDITIONING`s), insert **Sync Barrier**.
3. Wire `in_1`…`in_N` from each branch; pull `out_1`…`out_N` into the matching next-phase nodes.
4. New empty sockets appear automatically as you connect — you never scroll past hundreds of unused pins.
5. (Optional) After a barrier, drop **Phase Unload** on a passthrough wire before the next heavy model loads.

## Split phases across workflows

You do not have to keep CLIP, UNet, and VAE in one graph. Checkpoint between phases:

```text
1) CLIP only
   CLIPTextEncode → Save Conditioning  →  output/conditionings/

2) Diffusion only
   Load Conditioning → KSampler → Save Latent  →  output/latents/

3) VAE only
   Load Latent (Upload) → VAE Decode → Save Image
```

**Save Conditioning** writes a `.conditioning` file (passthrough so you can still wire a barrier in the same graph). **Load Conditioning** and **Load Latent (Upload)** each have a **choose file to upload** button (Comfy’s built-in Load Image upload only works for images; stock Load Latent has none). Uploads go to `input/conditionings/` or `input/latents/`. You can also use the dropdown or paste a path. Use distinct `filename_prefix` values per prompt (e.g. `conditionings/prompt_01`, `latents/zimage_01`).

That way each workflow only loads the model it needs — no Sync Barrier required *between* workflows, because the handoff is files on disk. Barriers still help when you batch many branches *inside* one workflow.

Conditioning save/load is inspired by the idea behind [ComfyUI-SaveAndLoadPromptCondition](https://github.com/endman100/ComfyUI-SaveAndLoadPromptCondition) (not on the Registry); this pack ships a clean MIT implementation aimed at phase-split batching.

## Example workflows

### One-shot / barrier batching

| File | Role |
|------|------|
| [`workflows/Z-Image-Sync-Barrier-Example.json`](workflows/Z-Image-Sync-Barrier-Example.json) | Full pipeline in one graph: CLIP barrier → sample → UNet barrier → VAE → Save Image |
| [`workflows/Z-Image-Sample-And-Checkpoint-Latents.json`](workflows/Z-Image-Sample-And-Checkpoint-Latents.json) | **Sample + checkpoint:** CLIP/UNet barriers, and `Save Latent` on each branch *before* the UNet barrier |
| [`workflows/Z-Image-Decode-Checkpointed-Latents.json`](workflows/Z-Image-Decode-Checkpointed-Latents.json) | **Finalize:** `Load Latent` → VAE Decode → Save Image (no samplers) |

### Split phase pipelines (CLIP → diffusion → VAE)

Run these in order. Each stage only needs its own model in memory.

| File | Role |
|------|------|
| [`workflows/Z-Image-Phase-CLIP-Save-Conditioning.json`](workflows/Z-Image-Phase-CLIP-Save-Conditioning.json) | **CLIP:** encode prompts → `Save Conditioning` |
| [`workflows/Z-Image-Phase-Diffusion-Save-Latents.json`](workflows/Z-Image-Phase-Diffusion-Save-Latents.json) | **Diffusion:** `Load Conditioning` → KSampler → `Save Latent` |
| [`workflows/Z-Image-Phase-VAE-Save-Images.json`](workflows/Z-Image-Phase-VAE-Save-Images.json) | **VAE:** `Load Latent (Upload)` → VAE Decode → Save Image |

### Why latent checkpoints exist (inside one sample workflow)

A UNet **Sync Barrier** waits until *every* sampler finishes before any VAE/Save Image can run. That is what stops model thrash — and it also means **no PNGs on disk until the whole sample phase is done**. If ComfyUI crashes mid-batch, finished branches only existed as in-memory latents.

```text
KSampler_i ──► Save Latent_i ──► UNet Barrier ──► (later) VAE / Save Image
```

Each `Save Latent` depends only on its own sampler, so it writes as soon as that branch finishes. Prefer distinct prefixes (e.g. `latents/zimage_01`). Decode later with the finalize / VAE-phase workflow — no resampling.

## Publishing / updates

This pack publishes to the Comfy Registry via GitHub Actions whenever `version` in `pyproject.toml` changes on `main`. Maintainers: set the repo secret `REGISTRY_ACCESS_TOKEN` to your Registry API key ([docs](https://docs.comfy.org/registry/publishing)).

## License

[MIT](LICENSE) — free to use, modify, and redistribute. Please keep the copyright notice and credit **josetseph**.

## GitHub repo tips (maintainers)

- **Description:** `Stop ComfyUI from reloading CLIP/UNet/VAE between branches when batching multiple prompts in one workflow.`
- **Topics:** `comfyui`, `comfyui-custom-nodes`, `model-management`, `vram-optimization`
