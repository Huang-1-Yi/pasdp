#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def pick_metric_key(payload: dict, preferred: Optional[str]) -> str:
	if preferred:
		if preferred not in payload:
			raise KeyError(f"指定的 metric key 不存在: {preferred}")
		return preferred

	for key in ["per_pos_nmse", "per_pos_mse", "per_pos_error"]:
		if key in payload:
			return key

	candidates = [k for k, v in payload.items() if isinstance(v, list) and len(v) > 1]
	if not candidates:
		raise ValueError("JSON 中未找到可绘制的按位置数组。")
	return candidates[0]


def format_metric_label(metric_key: str) -> str:
	"""将字段名转换为更易读的图例名。"""
	m = metric_key.lower()
	if m == "per_pos_mse":
		return "Per-position MSE (lower is better)"
	if m == "per_pos_nmse":
		return "Per-position NMSE (lower is better)"
	if m == "per_pos_gt_abs":
		return "Per-position |GT| mean"
	if m == "per_pos_gt_l2":
		return "Per-position ||GT||_2 mean"
	if m == "per_pos_gt_power":
		return "Per-position GT power mean"
	return metric_key


def find_metric_source_json(input_path: Path, preferred: Optional[str]) -> Path:
	"""
	兼容不同导出习惯：
	- 若 input_path 本身就包含可绘制数组，直接使用
	- 否则在同目录尝试常见结果文件（per_pos_mse.json / eval_2_log.json）
	"""
	candidates = [input_path]
	if input_path.name != "per_pos_mse.json":
		candidates.append(input_path.with_name("per_pos_mse.json"))
	if input_path.name != "eval_2_log.json":
		candidates.append(input_path.with_name("eval_2_log.json"))

	for p in candidates:
		if not p.exists():
			continue
		try:
			payload = json.loads(p.read_text(encoding="utf-8"))
			_ = pick_metric_key(payload, preferred)
			return p
		except Exception:
			continue

	raise ValueError(
		f"未在 {input_path.parent} 找到可绘制的 per-position 指标文件。"
	)


def main():
	parser = argparse.ArgumentParser(description="绘制 eval_2 JSON 的按位置指标曲线")
	parser.add_argument("json_path", type=str, help="输入 JSON 路径")
	parser.add_argument("-o", "--output", type=str, default=None, help="输出图片路径（默认同目录同名 .png）")
	parser.add_argument("--metric-key", type=str, default=None, help="要绘制的字段名，如 per_pos_mse / per_pos_nmse")
	parser.add_argument("--all", action="store_true", help="绘制所有 per_pos_* 数组到同一张图")
	parser.add_argument("--show", action="store_true", help="是否弹窗显示")
	parser.add_argument("--title", type=str, default=None, help="图标题")
	args = parser.parse_args()

	json_path = Path(args.json_path)
	if not json_path.exists():
		raise FileNotFoundError(f"文件不存在: {json_path}")

	source_json = find_metric_source_json(json_path, args.metric_key)
	payload = json.loads(source_json.read_text(encoding="utf-8"))

	if args.all:
		metric_keys = [k for k, v in payload.items() if k.startswith("per_pos_") and isinstance(v, list) and len(v) > 1]
		if not metric_keys:
			raise ValueError("当前 JSON 中没有可绘制的 per_pos_* 数组。")
	else:
		metric_keys = [pick_metric_key(payload, args.metric_key)]

	# 用第一条曲线长度作为 x 轴
	y0 = np.asarray(payload[metric_keys[0]], dtype=float)
	x = np.arange(len(y0), dtype=int)

	out_path = Path(args.output) if args.output else json_path.with_suffix(".png")

	fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
	for metric_key in metric_keys:
		y = np.asarray(payload[metric_key], dtype=float)
		m = min(len(x), len(y))
		ax.plot(x[:m], y[:m], marker="o", markersize=3.0, linewidth=1.3, label=format_metric_label(metric_key))
	ax.set_xlabel("Prediction position index t (0 = earliest, H-1 = latest)")
	if args.all:
		ax.set_ylabel("Metric value (mixed units)")
	else:
		mk = metric_keys[0].lower()
		if "mse" in mk or "error" in mk:
			ax.set_ylabel("Prediction error (lower is better)")
		elif "acc" in mk:
			ax.set_ylabel("Prediction accuracy (higher is better)")
		else:
			ax.set_ylabel("Metric value")
	ax.grid(True, alpha=0.25)

	# 如果是误差（MSE/NMSE），附加一个“精度”参考曲线：accuracy = 1 / (1 + error)
	if (not args.all) and ("mse" in metric_keys[0].lower() or "error" in metric_keys[0].lower()):
		y = np.asarray(payload[metric_keys[0]], dtype=float)
		acc = 1.0 / (1.0 + y)
		ax2 = ax.twinx()
		ax2.plot(x, acc, linestyle="--", linewidth=1.2, alpha=0.8, label="Derived accuracy proxy = 1/(1+error)")
		ax2.set_ylabel("Derived accuracy proxy (higher is better)")
		ax2.set_ylim(0.0, 1.05)

	count = payload.get("count", None)
	h = payload.get("horizon", len(y0))
	if args.all:
		title = args.title or f"Per-position prediction-quality trends (all metrics) (H={h}, count={count})"
	else:
		title = args.title or f"Per-position prediction-quality trend: {format_metric_label(metric_keys[0])} (H={h}, count={count})"
	ax.set_title(title)

	# 合并图例
	handles, labels = [], []
	for a in fig.axes:
		h_i, l_i = a.get_legend_handles_labels()
		handles.extend(h_i)
		labels.extend(l_i)
	if handles:
		ax.legend(handles, labels, loc="best")

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(out_path))
	print(f"Using source JSON: {source_json}")
	print(f"Saved plot to: {out_path}")

	if args.show:
		plt.show()
	else:
		plt.close(fig)


if __name__ == "__main__":
	main()

