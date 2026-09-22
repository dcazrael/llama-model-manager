# llama-model-manager

Personal Linux CLI for managing explicit `llama.cpp` model presets, downloading and staging GGUF models from Hugging Face, launching `llama-server`, and running reproducible benchmarks.

The project is intentionally small: one Python program, one installer/updater, and a user-owned config file outside the repository.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/dcazrael/llama-model-manager/main/install.sh | sh
```

On first install it asks for:

- the program install directory (default `~/.local/share/llama-model-manager`)
- the command directory (default `~/.local/bin`)
- the model directory (default `~/Models`)

It then writes:

```text
~/.config/llama-model-manager/config.ini
```

The config is never overwritten by updates.

The installer creates these commands:

```text
~/.local/bin/model
~/.local/bin/llama-benchmark
```

## Update

After the initial install, update directly through the CLI:

```bash
model update
```

This runs the installed updater, fetches the current `main` version, syntax-checks it, and only then replaces the installed program. Existing configuration is retained.

The underlying updater can still be run directly:

```bash
~/.local/share/llama-model-manager/install.sh
```

You can also rerun the one-line installer command; it uses the existing config and therefore the existing install locations.

## Configuration

Show the resolved config:

```bash
model config
```

Edit it interactively:

```bash
model setup
```

If `install_dir` or `bin_dir` changes, rerun `install.sh` afterward so the configured location is used. The installer does not delete the old install directory automatically.

Default configuration:

```ini
[install]
install_dir = ~/.local/share/llama-model-manager
bin_dir = ~/.local/bin

[paths]
models_dir = ~/Models
presets_yaml = ~/llm-workbench/configs/presets.yaml
presets_ini = ~/.config/llama.cpp/model-presets.ini
llama_server_bin = ~/Applications/llama.cpp/build/bin/llama-server
state_dir = ~/.local/state/llama-model-launcher
benchmark_case = ~/.config/llama.cpp/benchmarks/flash-next-v1.json
benchmark_results = ~/.local/share/llama-benchmarks/results.jsonl

[server]
host = 0.0.0.0
port = 1919
```

The actual generated config contains expanded absolute paths.

Environment variables can override runtime paths without changing the config:

```text
LLAMA_MODELS_DIR
LLAMA_MODELS_YAML
LLAMA_MODELS_INI
LLAMA_SERVER_BIN
LLAMA_MODEL_STATE_DIR
LLAMA_BENCH_CASE
LLAMA_BENCH_RESULTS
LLAMA_SERVER_HOST
LLAMA_SERVER_PORT
LLAMA_MODEL_MANAGER_CONFIG
```

## Usage

List configured presets:

```bash
model --list
```

Choose one interactively with `fzf`:

```bash
model
```

Start an exact preset:

```bash
model --preset Qwen3.8-Flash-Next
```

Run it in the background:

```bash
model --preset Qwen3.8-Flash-Next --background
```

Inspect the resolved `llama-server` invocation:

```bash
model --preset Qwen3.8-Flash-Next --show
```

Download a multipart GGUF quantization:

```bash
model --download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --file 'IQ3_XXS/*.gguf' \
  --quant IQ3_XXS
```

The Hugging Face CLI's native download/reconstruction progress is shown directly. After download, local staging also reports shard, size, percentage, transfer rate, target path, and completion status. There is deliberately no one-hour wrapper timeout around large model downloads.

Run the benchmark CLI through the installed benchmark link:

```bash
llama-benchmark --preset Qwen3.8-Flash-Next --matrix
```

The original explicit invocation remains supported as well:

```bash
python llama-model-manager.py model --list
python llama-model-manager.py benchmark --preset Qwen3.8-Flash-Next --matrix
```

## Dependencies

Core:

- Linux
- Python 3
- `llama.cpp` / `llama-server`
- Hugging Face `hf` CLI for downloads

Used by specific features:

- `fzf` for interactive preset selection
- `nvidia-smi` for NVIDIA GPU benchmark telemetry
- `ss` for safe listener detection

The Python program itself uses only the standard library.

## Safety behavior

The installer does not overwrite `config.ini` during updates. If `~/.local/bin/model` or `llama-benchmark` already exists as a regular file rather than a symlink, the installer asks before replacing it and otherwise stops.

Model downloads are delegated to the Hugging Face CLI. The manager does not impose its own total download timeout.
