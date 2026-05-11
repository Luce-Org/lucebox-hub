# Gemma 4 31B on RTX 4090

This Lucebox path serves `gemma-4-31B-it-abliterated-Q4_K_M.gguf` through the
Gemma 4 MTP-enabled llama.cpp backend. The DFlash runtime in this repository is
still the Qwen/Laguna research path; Gemma 4 uses libllama because it needs the
Gemma 4 graph, tokenizer template, and MTP assistant support.

Default workstation paths:

```bash
LUCEBOX_GEMMA4_MODEL=/mnt/c/Users/adyba/Downloads/gemma-4-31B-it-abliterated-Q4_K_M.gguf
LUCEBOX_GEMMA4_MTP_MODEL=/home/tdamre/models/gemma-4-31B-it-assistant-mtp-f16.gguf
LUCEBOX_LLAMA_SERVER=/home/tdamre/src/llama.cpp-mtp-pr22673/build-mtp-cuda124-speed-faall/bin/llama-server
```

Start from Windows PowerShell:

```powershell
.\scripts\Start-LuceboxGemma4090.ps1 -Command Start
```

Or from WSL:

```bash
./scripts/lucebox-gemma4-4090.sh start
```

The server listens on `http://127.0.0.1:18191` by default and exposes
OpenAI-compatible `/v1/chat/completions` plus llama.cpp `/completion`.
The launcher sets `--reasoning off` so OpenAI chat replies populate
`message.content` by default. It also pins `--spec-draft-n-max 4`, which is the
measured stable MTP window for this 31B target on the RTX 4090.

Verify the reply path and single-stream decode floor:

```bash
python3 scripts/verify_gemma4_4090.py --base-url http://127.0.0.1:18191 --threshold 60
```
