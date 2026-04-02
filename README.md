![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Fine-Tuning Blueprint</h1>

<p align="center"><em>Decentralized model fine-tuning on <a href="https://tangle.tools">Tangle</a> — operators run LoRA, QLoRA, and full fine-tuning jobs via axolotl, unsloth, or torchtune backends.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to run fine-tuning jobs with anonymous payments through shielded credits. Operators connect to local training servers (axolotl, unsloth, torchtune) or remote APIs (Modal, Lambda, RunPod) and register on-chain with GPU capabilities and supported methods.

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable fine-tuning results on Tangle
- **x402 HTTP** — fast private fine-tuning at `/v1/fine_tuning/jobs`

OpenAI Fine-tuning API compatible. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) with TEE support.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — wraps training backend, HTTP server, SpendAuth billing |
| `contracts/` | Solidity | FinetuneBSM — GPU validation, per-epoch pricing, method/model tracking |

## Pricing

Per-epoch pricing scaled by model size tier (parameters). Configured by the Blueprint admin via `configureModel()`. Training is priced per compute: `epochs * model_size_tier * price_per_epoch`.

## TEE Support

Add `features = ["tee"]` to `blueprint-sdk` in Cargo.toml. The `TeeLayer` middleware transparently attaches attestation metadata when running in a Confidential VM (H100 CC, SEV-SNP, TDX). Passes through when no TEE is configured.

## Quick Start

```bash
# Configure
cp config/operator.example.toml config/operator.toml
# Edit: supported models, methods, GPU specs, training endpoint URL, pricing

# Build
cargo build --release

# Run (requires a running training server)
FINETUNE_ENDPOINT=http://localhost:9000 ./target/release/finetune-operator
```

## Related Repos

- [Blueprint SDK](https://github.com/tangle-network/blueprint) — framework for building Blueprints
- [vLLM Inference Blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — text inference
- [Image Generation Blueprint](https://github.com/tangle-network/image-gen-inference-blueprint) — image generation
- [Voice Inference Blueprint](https://github.com/tangle-network/voice-inference-blueprint) — TTS/STT
- [Embedding Blueprint](https://github.com/tangle-network/embedding-inference-blueprint) — text embeddings
- [Video Generation Blueprint](https://github.com/tangle-network/video-gen-inference-blueprint) — video generation
