#!/usr/bin/env sh
set -eu

REPO="dcazrael/llama-model-manager"
RAW_BASE="${LLAMA_MODEL_MANAGER_RAW_BASE:-https://raw.githubusercontent.com/$REPO/main}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$CONFIG_HOME/llama-model-manager"
CONFIG_FILE="${LLAMA_MODEL_MANAGER_CONFIG:-$CONFIG_DIR/config.ini}"

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

prompt() {
    label="$1"
    default="$2"
    if [ -r /dev/tty ]; then
        printf '%s [%s]: ' "$label" "$default" > /dev/tty
        IFS= read -r answer < /dev/tty || answer=""
        printf '%s' "${answer:-$default}"
    else
        printf '%s' "$default"
    fi
}

confirm_replace() {
    path="$1"
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then
        return 0
    fi
    if [ -L "$path" ]; then
        return 0
    fi
    if [ -r /dev/tty ]; then
        printf '%s already exists and is not a symlink. Replace it? [y/N]: ' "$path" > /dev/tty
        IFS= read -r answer < /dev/tty || answer=""
        case "$answer" in
            y|Y|yes|YES) return 0 ;;
        esac
    fi
    fail "refusing to replace existing file: $path"
}

need python3
if command -v curl >/dev/null 2>&1; then
    FETCH="curl"
elif command -v wget >/dev/null 2>&1; then
    FETCH="wget"
else
    fail "curl or wget is required"
fi

mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    say "LLAMA MODEL MANAGER SETUP"
    say ""
    install_dir=$(prompt "Install directory" "$HOME/.local/share/llama-model-manager")
    bin_dir=$(prompt "Command directory" "$HOME/.local/bin")
    models_dir=$(prompt "Models directory" "$HOME/Models")

    INSTALL_DIR="$install_dir" BIN_DIR="$bin_dir" MODELS_DIR="$models_dir" CONFIG_FILE="$CONFIG_FILE" python3 - <<'PY'
import configparser
import os
from pathlib import Path

home = Path.home()
config_file = Path(os.environ["CONFIG_FILE"]).expanduser()
parser = configparser.ConfigParser(interpolation=None)
parser["install"] = {
    "install_dir": str(Path(os.environ["INSTALL_DIR"]).expanduser()),
    "bin_dir": str(Path(os.environ["BIN_DIR"]).expanduser()),
}
parser["paths"] = {
    "models_dir": str(Path(os.environ["MODELS_DIR"]).expanduser()),
    "presets_yaml": str(home / "llm-workbench/configs/presets.yaml"),
    "presets_ini": str(home / ".config/llama.cpp/model-presets.ini"),
    "llama_server_bin": str(home / "Applications/llama.cpp/build/bin/llama-server"),
    "state_dir": str(home / ".local/state/llama-model-launcher"),
    "benchmark_case": str(home / ".config/llama.cpp/benchmarks/flash-next-v1.json"),
    "benchmark_results": str(home / ".local/share/llama-benchmarks/results.jsonl"),
}
parser["server"] = {"host": "0.0.0.0", "port": "1919"}
config_file.parent.mkdir(parents=True, exist_ok=True)
tmp = config_file.with_suffix(config_file.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as handle:
    parser.write(handle)
os.replace(tmp, config_file)
PY
    say ""
    say "Created $CONFIG_FILE"
else
    say "Using existing configuration: $CONFIG_FILE"
fi

CONFIG_FILE="$CONFIG_FILE" python3 - <<'PY' > "$CONFIG_DIR/.install-paths"
import configparser
import os
from pathlib import Path

config_file = Path(os.environ["CONFIG_FILE"]).expanduser()
parser = configparser.ConfigParser(interpolation=None)
parser.read(config_file, encoding="utf-8")
print(Path(parser.get("install", "install_dir")).expanduser())
print(Path(parser.get("install", "bin_dir")).expanduser())
PY

INSTALL_DIR=$(sed -n '1p' "$CONFIG_DIR/.install-paths")
BIN_DIR=$(sed -n '2p' "$CONFIG_DIR/.install-paths")
rm -f "$CONFIG_DIR/.install-paths"

[ -n "$INSTALL_DIR" ] || fail "install_dir is empty in $CONFIG_FILE"
[ -n "$BIN_DIR" ] || fail "bin_dir is empty in $CONFIG_FILE"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

fetch_to() {
    url="$1"
    dest="$2"
    if [ "$FETCH" = "curl" ]; then
        curl -fsSL "$url" -o "$dest"
    else
        wget -qO "$dest" "$url"
    fi
}

say "Fetching latest llama-model-manager..."
fetch_to "$RAW_BASE/llama-model-manager.py" "$TMP_DIR/llama-model-manager.py"
fetch_to "$RAW_BASE/install.sh" "$TMP_DIR/install.sh"

python3 -m py_compile "$TMP_DIR/llama-model-manager.py" || fail "downloaded Python file failed syntax check"

mkdir -p "$INSTALL_DIR" "$BIN_DIR"
install -m 0755 "$TMP_DIR/llama-model-manager.py" "$INSTALL_DIR/llama-model-manager.py"
install -m 0755 "$TMP_DIR/install.sh" "$INSTALL_DIR/install.sh"

MODEL_LINK="$BIN_DIR/model"
BENCH_LINK="$BIN_DIR/llama-benchmark"
confirm_replace "$MODEL_LINK"
confirm_replace "$BENCH_LINK"
ln -sfn "$INSTALL_DIR/llama-model-manager.py" "$MODEL_LINK"
ln -sfn "$INSTALL_DIR/llama-model-manager.py" "$BENCH_LINK"

say ""
say "Installed llama-model-manager"
say "  program: $INSTALL_DIR/llama-model-manager.py"
say "  command: $MODEL_LINK"
say "  config:  $CONFIG_FILE"
say "  updater: $INSTALL_DIR/install.sh"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        say ""
        say "Note: $BIN_DIR is not currently in PATH."
        say "Add it to your shell configuration, for example:"
        say "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

say ""
say "Use 'model config' to inspect settings or 'model setup' to change them."
say "Run 'model update' later to install the latest repo version."
