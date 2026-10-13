#!/usr/bin/env bash
set -Eeuo pipefail

QUEUE_ID="${1:?usage: $0 q0|q0_remaining|q2|q1a|q1b|native_padp|native_transformer|native_padp_transformer}"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# CUDA compute ordinal 与 EGL render-device ordinal 是两套编号。
# GPU_UUID 使用 nvidia-smi -L 中的稳定 UUID（推荐）；PHYSICAL_GPU 仅作
# 兼容入口。这里通过 PCI bus 动态解决 CUDA PCI 顺序和 EGL 顺序。
configure_target_gpu() {
    local selector=$1
    local selector_kind=$2

    local query_result nvidia_index gpu_uuid pci_bus
    if [[ "$selector_kind" == "uuid" ]]; then
        if ! [[ "$selector" =~ ^GPU-[[:xdigit:]-]+$ ]]; then
            echo "ERROR: GPU_UUID must look like GPU-xxxxxxxx-..." >&2
            return 2
        fi
        query_result="$(nvidia-smi \
            --query-gpu=index,uuid,pci.bus_id --format=csv,noheader \
            | awk -F',' -v wanted="$selector" '
                { for (i=1; i<=NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
                $2 == wanted { print $1 "," $2 "," $3; exit }
            ')"
    else
        if ! [[ "$selector" =~ ^[0-9]+$ ]]; then
            echo "ERROR: PHYSICAL_GPU must be a nvidia-smi integer index" >&2
            return 2
        fi
        query_result="$(nvidia-smi \
            --query-gpu=index,uuid,pci.bus_id --format=csv,noheader \
            --id="$selector" | head -n 1 | tr -d ' ')"
    fi
    if [[ -z "$query_result" ]]; then
        echo "ERROR: cannot resolve target GPU: ${selector}" >&2
        return 2
    fi
    IFS=',' read -r nvidia_index gpu_uuid pci_bus <<< "$query_result"
    pci_bus="$(tr '[:upper:]' '[:lower:]' <<< "$pci_bus" | tr -d ' ')"
    if [[ -z "$pci_bus" ]]; then
        echo "ERROR: cannot resolve PCI bus for target GPU ${selector}" >&2
        return 2
    fi

    local cuda_pci_index
    cuda_pci_index="$(
        nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader \
        | tr '[:upper:]' '[:lower:]' | tr -d ' ' | sort \
        | awk -v wanted="$pci_bus" '$0 == wanted { print NR - 1; exit }'
    )"
    if [[ -z "$cuda_pci_index" ]]; then
        echo "ERROR: cannot resolve CUDA PCI ordinal for ${pci_bus}" >&2
        return 2
    fi

    local pci_path render_path render_node render_minor egl_index
    # nvidia-smi: 00000000:81:00.0; /dev/dri/by-path: 0000:81:00.0
    pci_bus="${pci_bus#00000000:}"
    pci_path="0000:${pci_bus}"
    render_path="/dev/dri/by-path/pci-${pci_path}-render"
    if [[ ! -L "$render_path" ]]; then
        echo "ERROR: EGL render-node link not found: ${render_path}" >&2
        return 2
    fi
    render_node="$(basename "$(readlink -f "$render_path")")"
    render_minor="${render_node#renderD}"
    if ! [[ "$render_minor" =~ ^[0-9]+$ ]] || ((render_minor < 128)); then
        echo "ERROR: unexpected DRM render node: ${render_node}" >&2
        return 2
    fi
    egl_index=$((render_minor - 128))

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    # robosuite 1.4.1 要求 MUJOCO_EGL_DEVICE_ID 同时出现在
    # CUDA_VISIBLE_DEVICES 中。目标 compute GPU 永远放第一个，
    # 因此项目配置里的 cuda:0 仍是目标物理卡。
    if [[ "$egl_index" == "$cuda_pci_index" ]]; then
        export CUDA_VISIBLE_DEVICES="$cuda_pci_index"
    else
        export CUDA_VISIBLE_DEVICES="${cuda_pci_index},${egl_index}"
    fi
    export MUJOCO_EGL_DEVICE_ID="$egl_index"

    echo "nvidia_index=${nvidia_index} gpu_uuid=${gpu_uuid} pci=${pci_path} "\
"render_node=${render_node} egl_index=${egl_index} "\
"cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
}

if [[ -n "${GPU_UUID:-}" && -n "${PHYSICAL_GPU:-}" ]]; then
    echo "ERROR: set only one of GPU_UUID or PHYSICAL_GPU" >&2
    exit 2
elif [[ -n "${GPU_UUID:-}" ]]; then
    configure_target_gpu "$GPU_UUID" uuid
elif [[ -n "${PHYSICAL_GPU:-}" ]]; then
    configure_target_gpu "$PHYSICAL_GPU" index
else
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
    echo "WARNING: GPU_UUID/PHYSICAL_GPU is unset; compute and EGL may select different hardware" >&2
fi

if [[ "${GPU_MAPPING_ONLY:-0}" == "1" ]]; then
    exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_NAME="${TASK_NAME:-square_d0}"
N_DEMO="${N_DEMO:-200}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs}"
RUNNER="${RUNNER:-robomimic/standard}"
LOG_ROOT="run_logs/ablation/robomimic/${TASK_NAME}"

mkdir -p "$LOG_ROOT"

CURRENT_PID=""
terminate_queue() {
    echo "[$(date '+%F %T')] TERMINATE queue=${QUEUE_ID} child=${CURRENT_PID:-none}"
    if [[ -n "${CURRENT_PID:-}" ]]; then
        kill -TERM "$CURRENT_PID" 2>/dev/null || true
        wait "$CURRENT_PID" 2>/dev/null || true
    fi
    exit 143
}
trap terminate_queue TERM INT

DATASET_PATH="data/robomimic/datasets/${TASK_NAME}/${TASK_NAME}_abs.hdf5"
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: dataset not found: $DATASET_PATH" >&2
    echo "Link server datasets first with:" >&2
    echo "  DP_DATA_ROOT=/media/disk7t/data bash tests/运行命令/链接数据集.sh" >&2
    exit 2
fi

common=(
    --config-name=train
    experiment=dp_robomimic
    "task_name=${TASK_NAME}"
    "n_demo=${N_DEMO}"
    encoder=rgb/robomimic_raw
    capacity=base
    horizon=16
    n_obs_steps=2
    n_action_steps=4
    "env_runner=${RUNNER}"
    "training.seed=${SEED}"
    training.resume=false
    training.pretrained_model_path=null
    training.early_stopping.enabled=false
)

run_controlled() {
    local label=$1
    local policy_config=$2
    shift 2

    local run_name="${TASK_NAME}_policy_${label}_raw_seed${SEED}"
    local run_log="${LOG_ROOT}/${label}_raw_seed${SEED}.log"
    echo "[$(date '+%F %T')] START controlled ${label} log=${run_log}"

    "$PYTHON_BIN" -u train_sim.py \
        "${common[@]}" \
        "policy=${policy_config}" \
        "$@" \
        "output_root=${OUTPUT_ROOT}" \
        "output_group=policy_ablation/controlled/${label}" \
        "name=${run_name}" \
        > "$run_log" 2>&1 &
    CURRENT_PID=$!
    echo "$CURRENT_PID" > "${run_log%.log}.pid"
    echo "[$(date '+%F %T')] CHILD controlled ${label} pid=${CURRENT_PID}"
    if wait "$CURRENT_PID"; then
        CURRENT_PID=""
        echo "[$(date '+%F %T')] DONE  controlled ${label}"
    else
        local rc=$?
        CURRENT_PID=""
        echo "[$(date '+%F %T')] FAILED controlled ${label} rc=${rc} log=${run_log}" >&2
        return "$rc"
    fi
}

run_native() {
    local label=$1
    local experiment=$2

    local run_name="${TASK_NAME}_native_${label}_seed${SEED}"
    local run_log="${LOG_ROOT}/${label}_native_seed${SEED}.log"
    echo "[$(date '+%F %T')] START native ${label} log=${run_log}"

    "$PYTHON_BIN" -u train_sim.py \
        --config-name=train \
        "experiment=${experiment}" \
        "task_name=${TASK_NAME}" \
        "n_demo=${N_DEMO}" \
        "env_runner=${RUNNER}" \
        "training.seed=${SEED}" \
        training.resume=false \
        training.pretrained_model_path=null \
        training.early_stopping.enabled=false \
        "output_root=${OUTPUT_ROOT}" \
        "output_group=policy_ablation/auxiliary/${label}" \
        "name=${run_name}" \
        > "$run_log" 2>&1 &
    CURRENT_PID=$!
    echo "$CURRENT_PID" > "${run_log%.log}.pid"
    echo "[$(date '+%F %T')] CHILD native ${label} pid=${CURRENT_PID}"
    if wait "$CURRENT_PID"; then
        CURRENT_PID=""
        echo "[$(date '+%F %T')] DONE  native ${label}"
    else
        local rc=$?
        CURRENT_PID=""
        echo "[$(date '+%F %T')] FAILED native ${label} rc=${rc} log=${run_log}" >&2
        return "$rc"
    fi
}

run_full_method() {
    local label=$1
    local experiment=$2

    local run_name="${TASK_NAME}_full_${label}_seed${SEED}"
    local run_log="${LOG_ROOT}/${label}_full_seed${SEED}.log"
    echo "[$(date '+%F %T')] START full-method ${label} log=${run_log}"

    # 完整方法实验只统一task、runner、seed和训练边界；不覆盖该
    # experiment自带的workspace、dataset、encoder、temporal和optimizer。
    "$PYTHON_BIN" -u train_sim.py \
        --config-name=train \
        "experiment=${experiment}" \
        "task_name=${TASK_NAME}" \
        "n_demo=${N_DEMO}" \
        "env_runner=${RUNNER}" \
        "training.seed=${SEED}" \
        training.resume=false \
        training.pretrained_model_path=null \
        training.early_stopping.enabled=false \
        "output_root=${OUTPUT_ROOT}" \
        "output_group=full_method/${label}" \
        "name=${run_name}" \
        > "$run_log" 2>&1 &
    CURRENT_PID=$!
    echo "$CURRENT_PID" > "${run_log%.log}.pid"
    echo "[$(date '+%F %T')] CHILD full-method ${label} pid=${CURRENT_PID}"
    if wait "$CURRENT_PID"; then
        CURRENT_PID=""
        echo "[$(date '+%F %T')] DONE  full-method ${label}"
    else
        local rc=$?
        CURRENT_PID=""
        echo "[$(date '+%F %T')] FAILED full-method ${label} rc=${rc} log=${run_log}" >&2
        return "$rc"
    fi
}

echo "project=${PROJECT_ROOT}"
echo "queue=${QUEUE_ID} gpu_uuid=${GPU_UUID:-unset} physical_gpu=${PHYSICAL_GPU:-unset} cuda_device_order=${CUDA_DEVICE_ORDER} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} mujoco_egl_device_id=${MUJOCO_EGL_DEVICE_ID:-unset}"
echo "python=$(command -v "$PYTHON_BIN") task=${TASK_NAME} seed=${SEED}"

case "$QUEUE_ID" in
    q0)
        run_controlled dp dp_ddim
        run_controlled streaming streaming_diffusion
        run_native bcrnn bcrnn_robomimic
        ;;
    q0_remaining)
        # q0 中的 DP 已完整跑完时使用；只从头补跑尚未完成的
        # Streaming Diffusion，完成后再运行 BCRNN。
        run_controlled streaming streaming_diffusion
        run_native bcrnn bcrnn_robomimic
        ;;
    q2)
        run_controlled padp padp_positionwise +window_exp_gamma=0.2
        run_controlled streaming_optimized streaming_diffusion_optimized
        ;;
    q1a)
        run_controlled focal focal_flow
        run_controlled diffusion_transformer diffusion_transformer
        ;;
    q1b)
        run_controlled flow flow_policy
        run_native act act_robomimic
        ;;
    native_padp_transformer)
        # PADP原生：H40/O1/A1、robomimic encoder、PADP workspace/dataset。
        run_full_method padp_native_h40_o1_a1 padp_robomimic
        # Diffusion Transformer原生：H16/O2/A8、专用optimizer。
        run_full_method diffusion_transformer_native_h16_o2_a8 \
            diffusion_transformer_robomimic
        ;;
    native_padp)
        run_full_method padp_native_h40_o1_a1 padp_robomimic
        ;;
    native_transformer)
        run_full_method diffusion_transformer_native_h16_o2_a8 \
            diffusion_transformer_robomimic
        ;;
    *)
        echo "ERROR: unknown queue ${QUEUE_ID}; expected q0, q0_remaining, q2, q1a, q1b, native_padp, native_transformer or native_padp_transformer" >&2
        exit 2
        ;;
esac

echo "[$(date '+%F %T')] ALL DONE queue=${QUEUE_ID}"
