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

## Example workflows

| File | Role |
|------|------|
| [`workflows/Z-Image-Sync-Barrier-Example.json`](workflows/Z-Image-Sync-Barrier-Example.json) | Full pipeline in one graph: CLIP barrier → sample → UNet barrier → VAE → Save Image |
| [`workflows/Z-Image-Sample-And-Checkpoint-Latents.json`](workflows/Z-Image-Sample-And-Checkpoint-Latents.json) | **Part 1 — sample + checkpoint:** CLIP/UNet barriers, and `Save Latent` on each branch *before* the UNet barrier |
| [`workflows/Z-Image-Decode-Checkpointed-Latents.json`](workflows/Z-Image-Decode-Checkpointed-Latents.json) | **Part 2 — finalize:** `Load Latent` → VAE Decode → Save Image (no samplers) |

### Why the two-part latent checkpoint approach exists

A UNet **Sync Barrier** waits until *every* sampler finishes before any VAE/Save Image can run. That is what stops model thrash — and it also means **no PNGs on disk until the whole sample phase is done**. If ComfyUI crashes mid-batch, finished branches only existed as in-memory latents.

The fix is a **workflow** pattern, not a change to the barrier node:

```text
KSampler_i ──► Save Latent_i ──► UNet Barrier ──► (later) VAE / Save Image
```

Each `Save Latent` depends only on its own sampler, so it writes to disk as soon as that branch finishes. The barrier still gates VAE so you keep one clean UNet phase.

Use **Part 1** for the heavy CLIP+UNet run. If anything fails afterward (or you want to decode later / on another machine), run **Part 2**: point each `Load Latent` at the files under `output/latents/`, decode with the same VAE, and save images — no need to resample.

Prefer distinct `filename_prefix` values per branch (e.g. `latents/zimage_01`) so resume mapping stays obvious.

## Publishing / updates

This pack publishes to the Comfy Registry via GitHub Actions whenever `version` in `pyproject.toml` changes on `main`. Maintainers: set the repo secret `REGISTRY_ACCESS_TOKEN` to your Registry API key ([docs](https://docs.comfy.org/registry/publishing)).

## License

[MIT](LICENSE) — free to use, modify, and redistribute. Please keep the copyright notice and credit **josetseph**.

## GitHub repo tips (maintainers)

- **Description:** `Stop ComfyUI from reloading CLIP/UNet/VAE between branches when batching multiple prompts in one workflow.`
- **Topics:** `comfyui`, `comfyui-custom-nodes`, `model-management`, `vram-optimization`
