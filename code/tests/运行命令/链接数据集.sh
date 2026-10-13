#!/usr/bin/env bash
set -u

# 将集中存放的数据映射到项目沿用的 data/... 路径。
# 显式 DP_DATA_ROOT 优先；服务器数据盘其次；最后兼容原本的家目录布局。
if [[ -n "${DP_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$DP_DATA_ROOT"
elif [[ -d /media/disk7t/data ]]; then
  DATA_ROOT=/media/disk7t/data
else
  DATA_ROOT=/home/hy/data
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DATA="${PROJECT_ROOT}/data"

errors=0
created=0
kept=0

link_one() {
  local source="$1"
  local target="$2"

  if [[ ! -e "$source" ]]; then
    echo "[缺少源数据] $source" >&2
    errors=$((errors + 1))
    return
  fi

  if [[ -L "$target" ]]; then
    local current
    current="$(readlink -f -- "$target" 2>/dev/null || true)"
    if [[ "$current" == "$(readlink -f -- "$source")" ]]; then
      echo "[已存在] $target -> $(readlink -- "$target")"
      kept=$((kept + 1))
    else
      echo "[拒绝覆盖] $target 已指向 $(readlink -- "$target")" >&2
      errors=$((errors + 1))
    fi
    return
  fi

  if [[ -e "$target" ]]; then
    echo "[拒绝覆盖] $target 已是实际文件或目录" >&2
    errors=$((errors + 1))
    return
  fi

  ln -s -- "$source" "$target"
  echo "[新建] $target -> $source"
  created=$((created + 1))
}

mkdir -p -- "$PROJECT_DATA"

# 将数据根目录下的一级数据集目录全部映射到项目 data/，与 outputs 同级。
# 跳过 output/outputs，避免将另一套实验输出树链接进当前项目。
shopt -s nullglob
dataset_roots=("${DATA_ROOT}"/*)
for source in "${dataset_roots[@]}"; do
  [[ -d "$source" ]] || continue
  name="$(basename -- "$source")"
  case "$name" in
    output|outputs) continue ;;
  esac
  link_one "$source" "${PROJECT_DATA}/${name}"
done

# MetaWorld / Adroit 的现有 Hydra 配置使用 data/<数据集名>.zarr，
# 因此逐项创建平铺链接；新增同名 Zarr 后再次运行本脚本即可自动接入。
metaworld_sources=("${DATA_ROOT}/metaworld"/metaworld_*_expert.zarr)
adroit_sources=("${DATA_ROOT}/adroit"/adroit_*_expert.zarr)

if ((${#metaworld_sources[@]} == 0)); then
  echo "[缺少源数据] ${DATA_ROOT}/metaworld/metaworld_*_expert.zarr" >&2
  errors=$((errors + 1))
else
  for source in "${metaworld_sources[@]}"; do
    link_one "$source" "${PROJECT_DATA}/$(basename -- "$source")"
  done
fi

if ((${#adroit_sources[@]} == 0)); then
  echo "[缺少源数据] ${DATA_ROOT}/adroit/adroit_*_expert.zarr" >&2
  errors=$((errors + 1))
else
  for source in "${adroit_sources[@]}"; do
    link_one "$source" "${PROJECT_DATA}/$(basename -- "$source")"
  done
fi

echo "完成：新建 ${created}，已存在 ${kept}，错误 ${errors}。"
if ((errors > 0)); then
  exit 1
fi
