#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv=$(realpath -m "${VENV:-$root/.venv-pex88096}")
wheel_dir=$(realpath -m "${WHEEL_DIR:-$root/wheels}")
patches="$root/patches/pex88096"
overlay="$root/overlays/pex88096-vision14"
release=https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision14-sm89-sm120-cu130
vllm_wheel='vllm-0.33.0+glm53.dsv41.vision14.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl'
vllm_sha=8f400eb43378c4ce8191df298de344e559d32724b190376509ce49a3e10ae675

case "${FLASHINFER_VARIANT:-post1}" in
  post1)
    fi_wheel='flashinfer_python-0.7.3+glm53.dsv41.vision4.sm89sm120.cu130.pt214.post1-py3-none-any.whl'
    fi_sha=af2938d1155318c2eed8db832d485759905ca300280707f6829a7f80c827a214
    ;;
  original)
    fi_wheel='flashinfer_python-0.7.3+glm53.dsv41.vision4.sm89sm120.cu130.pt214-py3-none-any.whl'
    fi_sha=d7cf9ee42ca8a86ed2fc7d0615abec0655fac000962a90979c8e0d4d860d9ae0
    if [[ ! -f "$wheel_dir/$fi_wheel" ]]; then
      echo "Supply the original vision4 wheel in $wheel_dir or use FLASHINFER_VARIANT=post1." >&2
      exit 1
    fi
    ;;
  *)
    echo 'FLASHINFER_VARIANT must be post1 or original.' >&2
    exit 1
    ;;
esac

command -v uv >/dev/null || { echo 'Install uv before running setup.' >&2; exit 1; }
mkdir -p "$wheel_dir" "$root/overlays"
scratch=$(mktemp -d "$root/overlays/.setup.XXXXXX")
download=
cleanup() {
  [[ -z "$download" ]] || rm -f -- "$download"
  rm -rf -- "$scratch"
}
trap cleanup EXIT

check_hash() {
  printf '%s  %s\n' "$2" "$1" | sha256sum --check --status
}

prepare_wheel() {
  local name=$1 hash=$2 target="$wheel_dir/$1"
  if [[ ! -f "$target" ]]; then
    download=$(mktemp "$wheel_dir/.download.XXXXXX")
    curl --fail --location --retry 2 --connect-timeout 15 --silent --show-error \
      --output "$download" "$release/${name//+/%2B}"
    if ! check_hash "$download" "$hash"; then
      echo "Downloaded artifact failed SHA256 verification: $name" >&2
      exit 1
    fi
    mv -- "$download" "$target"
    download=
  fi
  if ! check_hash "$target" "$hash"; then
    echo "Artifact failed SHA256 verification: $target" >&2
    exit 1
  fi
}

prepare_wheel "$vllm_wheel" "$vllm_sha"
prepare_wheel "$fi_wheel" "$fi_sha"

if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.12 "$venv"
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$venv/bin/python" - "$wheel_dir/$fi_wheel" <<'PY' > "$scratch/flashinfer-override.txt"
from pathlib import Path
import sys

print("flashinfer-python @ " + Path(sys.argv[1]).resolve().as_uri())
PY
uv pip install --python "$venv/bin/python" \
  --index-url "${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --extra-index-url "${PYTORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu130}" \
  --index-strategy unsafe-best-match --constraint "$root/profiles/pex88096-constraints.txt" \
  --override "$scratch/flashinfer-override.txt" "$wheel_dir/$vllm_wheel" "$wheel_dir/$fi_wheel"
uv pip check --python "$venv/bin/python"
"$venv/bin/python" -c 'import torch; assert torch.version.cuda == "13.0", torch.version.cuda'

site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
if [[ -e "$overlay" ]]; then
  (cd "$overlay" && sha256sum --check "$patches/overlay.sha256")
else
  mkdir "$scratch/overlay"
  cp -a "$site/vllm" "$scratch/overlay/"
  for patch in prefill-indexer-sharding ada-prefill-tiles decode-indexer-sharding; do
    patch --directory "$scratch/overlay" --batch --forward --fuzz=0 -p1 < "$patches/$patch.patch"
  done
  (cd "$scratch/overlay" && sha256sum --check "$patches/overlay.sha256")
  mv -T --no-clobber "$scratch/overlay" "$overlay"
  (cd "$overlay" && sha256sum --check "$patches/overlay.sha256")
fi
printf 'PEX88096 environment ready at %s\n' "$venv"
