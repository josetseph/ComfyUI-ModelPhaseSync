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

Search for **Model Phase Sync** (or `model-phase-sync`) after it is published on the [Comfy Registry](https://registry.comfy.org).

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

See [`workflows/Z-Image-Sync-Barrier-Example.json`](workflows/Z-Image-Sync-Barrier-Example.json) for a working Z-Image Turbo multi-prompt example.

## Publishing / updates

This pack publishes to the Comfy Registry via GitHub Actions whenever `version` in `pyproject.toml` changes on `main`. Maintainers: set the repo secret `REGISTRY_ACCESS_TOKEN` to your Registry API key ([docs](https://docs.comfy.org/registry/publishing)).

## License

[MIT](LICENSE) — free to use, modify, and redistribute. Please keep the copyright notice and credit **josetseph**.

## GitHub repo tips (maintainers)

- **Description:** `Stop ComfyUI from reloading CLIP/UNet/VAE between branches when batching multiple prompts in one workflow.`
- **Topics:** `comfyui`, `comfyui-custom-nodes`, `model-management`, `vram-optimization`
