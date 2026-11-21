#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_DIR/env.sh"

# The CUDA runtime wheel only ships a versioned soname, while FlashInfer's
# JIT linker requests the conventional unversioned development name.
if [[ ! -e "$CUDA_LIB_DIR/libcudart.so" ]]; then
  if [[ ! -e "$CUDA_LIB_DIR/libcudart.so.13" ]]; then
    echo "Missing libcudart.so and libcudart.so.13 in $CUDA_LIB_DIR" >&2
    exit 1
  fi
  ln -s libcudart.so.13 "$CUDA_LIB_DIR/libcudart.so"
fi

mkdir -p \
  "$PYTORCH_KERNEL_CACHE_PATH" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"

MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/DeepSeek-V4.1-Flash-1017}"
"$PROJECT_DIR/tools/verify-model.sh" "$MODEL_DIR"

: "${HOST:=127.0.0.1}"
: "${PORT:=8011}"
: "${SERVED_MODEL_NAME:=deepseek-v4.1-flash}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MAX_MODEL_LEN:=262144}"
: "${MAX_NUM_SEQS:=1}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${ENABLE_CED:=1}"
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENABLE_DSPARK:=1}"
: "${NUM_SPECULATIVE_TOKENS:=3}"
: "${ENABLE_ADAPTIVE_VERIFICATION:=1}"
: "${ENABLE_VISION:=1}"
: "${MAX_IMAGES_PER_PROMPT:=2}"

if [[ "${PEX88096_MODE16:-0}" == "1" ]]; then
  if [[ "${PEX88096_PROFILE:-0}" != "1" ]]; then
    echo 'Mode 16 requires the PEX88096 profile.' >&2
    exit 1
  fi
  MAX_NUM_SEQS=8
  state="${MODE16_STATE_DIR:-$PROJECT_DIR/profiles/mode16}"
  for setting in mode ffn-mode scheduler-mode; do
    if [[ ! -s "$state/$setting" ]]; then
      echo "Missing mode 16 setting: $state/$setting" >&2
      exit 1
    fi
  done
  export DS1017_ADA_CED_SP_SWITCH="$state/mode"
  export DS1017_ADA_FFN_GATHER_SWITCH="$state/ffn-mode"
  export DS1017_MIXED_PREFILL_SWITCH="$state/scheduler-mode"
  export DS1017_ADA_PACK_EXPERT_SCALES=0
fi

if [[ "${PEX88096_MODE17:-0}" == "1" ]]; then
  : "${DS1017_STAGED_K:=1}"
  case "$DS1017_STAGED_K" in
    0|1) export DS1017_STAGED_K ;;
    *) echo 'DS1017_STAGED_K must be 0 or 1.' >&2; exit 1 ;;
  esac
fi

: "${NCCL_P2P_DISABLE:=1}"
export CUDA_VISIBLE_DEVICES NCCL_P2P_DISABLE

args=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 8
  --distributed-executor-backend mp
  --enable-expert-parallel
  --moe-backend auto
  --kv-cache-dtype fp8
  --block-size 128
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --engram-config '{"cpu_offload":true}'
  --load-format safetensors
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v41
)

if [[ "${PEX88096_PROFILE:-0}" == "1" ]]; then
  overlay="$PROJECT_DIR/overlays/pex88096-vision14"
  setup_args=
  if [[ "${PEX88096_MODE17:-0}" == "1" ]]; then
    overlay+="-mode17"
    setup_args=' --mode17'
  elif [[ "${PEX88096_MODE16:-0}" == "1" ]]; then
    overlay+="-mode16"
    setup_args=' --mode16'
  fi
  if [[ ! -x "$VIRTUAL_ENV/bin/vllm" || ! -s "$overlay/vllm/__init__.py" ]]; then
    echo "Run ./setup_pex88096.sh$setup_args before using this PEX88096 profile." >&2
    exit 1
  fi
  export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
  args+=(--safetensors-load-strategy prefetch --safetensors-prefetch-num-threads 2)
  args+=(--no-enable-extensible-kv-cache --prefill-schedule-interval 1)
  args+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":48}')
else
  args+=(--disable-custom-all-reduce --safetensors-load-strategy lazy)
fi

if [[ "$ENABLE_VISION" == "1" ]]; then
  if [[ ! "$MAX_IMAGES_PER_PROMPT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_IMAGES_PER_PROMPT must be a positive integer when ENABLE_VISION=1" >&2
    exit 1
  fi
  args+=(--limit-mm-per-prompt "{\"image\":$MAX_IMAGES_PER_PROMPT}")
else
  args+=(--language-model-only)
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  args+=(--enable-prefix-caching)
else
  args+=(--no-enable-prefix-caching)
fi

if [[ "$ENABLE_CED" == "1" ]]; then
  args+=(--hf-overrides '{"ced_prefill":true}')
fi

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  adaptive=false
  [[ "$ENABLE_ADAPTIVE_VERIFICATION" == "1" ]] && adaptive=true
  draft_tp=""
  if [[ "${PEX88096_PROFILE:-0}" == "1" ]]; then
    draft_tp=',"draft_tensor_parallel_size":8'
  fi
  args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$adaptive$draft_tp}")
fi

echo "Starting $SERVED_MODEL_NAME on GPUs $CUDA_VISIBLE_DEVICES"
echo "Context=$MAX_MODEL_LEN DSpark=$ENABLE_DSPARK/$NUM_SPECULATIVE_TOKENS CED=$ENABLE_CED prefix_cache=$ENABLE_PREFIX_CACHING vision=$ENABLE_VISION/$MAX_IMAGES_PER_PROMPT"
if [[ "${PEX88096_MODE17:-0}" == "1" ]]; then
  echo "Indexer K staging=$DS1017_STAGED_K"
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'vllm'
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi
exec vllm "${args[@]}"
