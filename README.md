<p align="center">
  <img src="assets/banner.png" alt="Lucebox" width="85%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e8e8ed?style=for-the-badge&labelColor=090909" alt="MIT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>Open LLM inference, rewritten by hand for one specific chip at a time.</strong><br/>
  Kernels, speculative decoding, and quantization, tailored per target.<br/>
  We don't wait for better silicon. We rewrite the software.
</p>

---

> The efficiency gap between consumer and datacenter silicon isn't inherent.
> It's an artifact of running generic software on capable hardware.

Local AI is becoming the default for developers shipping agents, researchers reproducing results, teams tired of per-token bills, and anyone whose data shouldn't leave the room. The hardware to run a 20B+ hybrid model at home mostly exists already, a second-hand RTX 3090, a Ryzen AI MAX+ 395, an Apple M-series chip. The kernels to run one *well* don't.

Lucebox is where we build them. One model, one hardware target, one optimization problem at a time. Open weights, open kernels, open benchmarks.

[Read the first writeup →](https://lucebox.com/blog/megakernel)

---

## Inside the box

Two projects today, more coming. Each one is a self-contained release with its own benchmarks and paper-style writeup.

<p align="center">
  <a href="megakernel/"><img src="assets/svg/card-megakernel-dark.svg" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="dflash/"><img src="assets/svg/card-dflash-dark.svg" alt="DFlash 27B" width="46%"></a>
</p>

---

## 01 · Megakernel

**The first megakernel for hybrid DeltaNet/Attention LLMs.** All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch, 1.87 tok/J on a 2020 GPU, matching Apple's latest silicon at 2× the throughput.

```bash
git clone https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/megakernel
pip install -e .
python final_bench.py
```

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **37,800** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | n/a |

**What makes it work:** 82 blocks, 512 threads, one persistent kernel. No CPU round-trips between layers. Weights streamed straight from HuggingFace. Cooperative grid sync instead of ~100 kernel launches per token. Power ceiling hit before compute ceiling, so DVFS converts tight execution straight into saved watts.

[Full writeup →](megakernel/README.md) · [Benchmarks →](megakernel/RESULTS.md) · [Blog post →](https://lucebox.com/blog/megakernel)

---

## 02 · DFlash 27B

**First open-source port of DFlash speculative decoding to consumer GPUs.** Qwen3.5-27B at 130 tok/s on a single RTX 3090 (Q4_K_M target + BF16 draft). 128K context in 24 GB. 3.5× faster than chain speculative decoding, 2.9× faster than SGLang AWQ on the same hardware.

```
                     AR (tok/s)   DFlash (tok/s)   Speedup
  HumanEval              37.4         130.7         3.49×
  Math500                37.4         111.2         2.97×
  GSM8K                  37.6          97.0         2.58×
```

**The constraint that shaped the project.** AWQ INT4 of Qwen3.5-27B plus the BF16 draft doesn't leave room for the DDTree verify state on a 24 GB card. Q4_K_M GGUF (14.9 GB target) is the largest format that fits target + 3.46 GB draft + budget=22 tree state + KV cache in 24 GB on the RTX 3090. Picking it forced a new port on top of ggml, since no public DFlash runtime supports a GGUF target.

**What we built vs what we didn't.** The algorithms are not ours:
- [**DFlash**](https://arxiv.org/abs/2502.20762) (z-lab, 2025): block-diffusion draft conditioned on target hidden states.
- [**DDTree**](https://arxiv.org/abs/2502.20762) (Ringel et al., 2025): tree-structured verify that beats chain verify at the same compute budget.

What we ported and tuned:
- C++/CUDA decode engine on top of ggml (no libllama, no Python runtime, Q4_K_M target path).
- Three custom CUDA kernels for tree-aware SSM state rollback: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`.
- DDTree budget swept for RTX 3090 + Q4_K_M target: **budget=22** is the sweet spot.
- Q4_0 KV cache + sliding `target_feat` ring to fit 128K context in 24 GB with ~3% AL hit.

[Full writeup →](dflash/README.md) · [Benchmarks →](dflash/RESULTS.md) · [Blog post →](https://lucebox.com/blog/dflash)

---

## Why this exists

Local AI should be a default, not a privilege. Running capable models on your own machine keeps your data private, skips the per-token bill, and removes the vendor dependency. The hardware to do it already sits on desks: an RTX 3090, a Ryzen AI MAX+ 395, an Apple M-series chip. The software to run those chips well doesn't.

General-purpose frameworks won the last decade for a reason: hand-tuning a kernel for every chip and every model was too expensive to justify. So everyone ships one stack that's decent on everything and great on nothing. Most of the silicon's capability stays on the floor.

That calculus is flipping. AI-assisted development makes per-chip, per-model rewrites tractable in days instead of months. A kernel that would have taken a small team a quarter now fits inside a single release cycle. Lucebox is where we publish those rewrites: one chip, one model family, one optimization problem at a time. Each artifact ships standalone, a kernel, a spec-decode port, a benchmark harness, with full writeup and MIT source. Together they build toward a single command surface, `lucebox`, that makes local inference on consumer chips a one-line install.

---

## Quickstart

```bash
git clone https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub

# megakernel (Qwen 3.5-0.8B, batch 1)
cd megakernel && pip install -e . && python final_bench.py

# dflash 27B                                                [coming soon]
```

**Requirements:** NVIDIA GPU (Ampere+), CUDA 12+, PyTorch 2.0+. Tested on RTX 3090 (2020).

**Optional, find your GPU's sweet spot:** `sudo nvidia-smi -pl 220`

---

## Repository layout

```
lucebox-hub/
├── megakernel/    · fused forward pass for Qwen 3.5-0.8B
├── dflash/        · DFlash speculative decoding port for Qwen 3.5-27B on RTX 3090
└── assets/        · banners, cards, diagrams
```

---

## Roadmap

```
  Q1 2026    ▮▮▮▮▮▮▮▮▮▮    RTX 3090 kernels & optimizations
  Q2 2026    ▮▮▮▮▮▯▯▯▯▯    Ryzen AI MAX+ 395 optimizations
  Q3 2026    ▮▮▯▯▯▯▯▯▯▯    heterogeneous CPU + GPU with unified memory
```

---

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub},
  year   = {2026}
}
```

Per-project citations live in each subproject's README.

---

## Inspired by

- [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): megakernel idea and the intelligence-per-watt methodology.
- [z-lab/DFlash](https://arxiv.org/abs/2502.20762) (Wang et al., 2025): block-diffusion speculative decoding algorithm. We use their published Qwen3.5-27B-DFlash draft weights as-is.
- [DDTree](https://arxiv.org/abs/2604.12989) (Ringel & Romano, 2025): tree-structured verify that DFlash 27B uses for its 3.5× speedup over chain spec decoding. [liranringel/ddtree](https://github.com/liranringel/ddtree).
- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen): prior art on fused Qwen kernels.

---

## Community

- **Discord**: [discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **Website**: [lucebox.com](https://lucebox.com)
- **Issues**: [github.com/Luce-Org/lucebox-hub/issues](https://github.com/Luce-Org/lucebox-hub/issues)
- **Blog**: [lucebox.com/blog](https://lucebox.com/blog)

---

<p align="center">
  <sub><a href="LICENSE">MIT</a> · <a href="https://lucebox.com">Lucebox.com</a></sub>
</p>
