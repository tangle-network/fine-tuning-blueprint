![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Fine-tuning Blueprint</h1>

<p align="center"><em>Decentralized model fine-tuning on <a href="https://tangle.tools">Tangle</a> — operators run LoRA, QLoRA, and full fine-tuning on GPU hardware.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to offer model fine-tuning as a service. Users upload training data and receive fine-tuned adapter weights. Operators compete on price, hardware quality, and turnaround time.

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable training on Tangle
- **x402 HTTP** — private fine-tuning at `/v1/fine_tuning/jobs`

OpenAI Fine-tuning API compatible. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) with TEE support.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — wraps training backend, async job management, SpendAuth billing |
| `contracts/` | Solidity | FinetuneBSM — per-epoch pricing, model tier validation, method support |

## Supported Methods

- **LoRA** — Low-Rank Adaptation, 24GB+ VRAM, fast training
- **QLoRA** — Quantized LoRA, 16GB+ VRAM, memory efficient
- **Full fine-tuning** — 80GB+ VRAM, highest quality

## Pricing

Per-epoch per model size tier. Blueprint admin configures tiers (e.g., 7B models = X per epoch, 70B = Y per epoch).

## TEE Support

Add `features = ["tee"]` to `blueprint-sdk` in Cargo.toml. Confidential fine-tuning — operator cannot see training data or resulting weights.

## Quick Start

```bash
# Local mode (requires axolotl/unsloth training server)
FINETUNE_ENDPOINT=http://localhost:8000 cargo run --release

# API mode (requires Modal/RunPod endpoint)
FINETUNE_ENDPOINT=https://your-modal-endpoint cargo run --release
```

## Related Repos

- [Blueprint SDK](https://github.com/tangle-network/blueprint) — framework for building Blueprints
- [vLLM Inference Blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — text inference
- [Voice Inference Blueprint](https://github.com/tangle-network/voice-inference-blueprint) — TTS/STT
- [Image Generation Blueprint](https://github.com/tangle-network/image-gen-inference-blueprint) — image generation
- [Embedding Blueprint](https://github.com/tangle-network/embedding-inference-blueprint) — text embeddings
- [Video Generation Blueprint](https://github.com/tangle-network/video-gen-inference-blueprint) — video generation
