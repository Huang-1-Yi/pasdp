#!/usr/bin/env python3
import json
from pathlib import Path
import math

PADP_AGG = Path('/media/disk7t/PADP1018/accurency_test/data/eval/test_padp/pmse/per_pos_mse_aggregated.json')
DP40_AGG = Path('/media/disk7t/PADP1018/accurency_test/data/eval/test_dp_h40/pmse/per_pos_mse_aggregated.json')

# Table order mapping (from the LaTeX table): C., T., P.P., C.P., H.C., Sq., St., S.T., N.A., T.P.A.
# Use more specific substrings to avoid ambiguous matches (e.g. 'stack' vs 'stack_three')
TASK_MATCH = [
    ('coffee', 'coffee_d0'),
    ('threading', 'threading_d0'),
    ('pick_place', 'pick_place_d0'),
    ('coffee_preparation', 'coffee_preparation_d0'),
    ('hammer_cleanup', 'hammer_cleanup_d0'),
    ('square', 'square_d0'),
    ('stack', 'stack_d0'),
    ('stack_three', 'stack_three_d0'),
    ('nut_assembly', 'nut_assembly_d0'),
    ('three_piece_assembly', 'three_piece_assembly_d0')
]

# Success rates from the LaTeX table (PADP and DP at H=40), in same order
PADP_success = [100, 90, 33, 91, 100, 97, 98, 43, 54, 64]
DP_success   = [96, 86, 17, 90, 100, 80, 100, 58, 56, 76]


def load_agg(path: Path):
    return json.loads(path.read_text())


def find_key(files_dict, match_substr):
    # Prefer keys that contain the more specific token (match_substr),
    # fall back to any key that contains the shorter token if specific not found.
    for k in files_dict.keys():
        if match_substr in k:
            return k
    # fallback: try splitting on '_' and match first token
    short = match_substr.split('_')[0]
    for k in files_dict.keys():
        if short in k:
            return k
    return None


def pearson(x, y):
    n = len(x)
    if n == 0:
        return float('nan')
    mx = sum(x)/n
    my = sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    denx = math.sqrt(sum((xi-mx)**2 for xi in x))
    deny = math.sqrt(sum((yi-my)**2 for yi in y))
    if denx==0 or deny==0:
        return float('nan')
    return num/(denx*deny)


def spearman(x, y):
    # compute ranks (avg rank for ties)
    def ranks(a):
        # returns list of ranks starting at 1
        sorted_idx = sorted(range(len(a)), key=lambda i: a[i])
        ranks = [0]*len(a)
        i = 0
        while i < len(a):
            j = i
            # find ties
            while j+1 < len(a) and a[sorted_idx[j+1]] == a[sorted_idx[i]]:
                j += 1
            # average rank for indices i..j
            avg_rank = (i + j + 2) / 2.0  # +1 for 1-based
            for k in range(i, j+1):
                ranks[sorted_idx[k]] = avg_rank
            i = j+1
        return ranks
    rx = ranks(list(x))
    ry = ranks(list(y))
    return pearson(rx, ry)


def main():
    padp = load_agg(PADP_AGG)
    dp40 = load_agg(DP40_AGG)
    padp_files = diffusion_policy.get('files', {})
    dp40_files = dp40.get('files', {})

    rows = []
    padp_mse0 = []
    dp40_mse0 = []
    padp_succ = []
    dp40_succ = []

    for i, (keyname, match) in enumerate(TASK_MATCH):
        padp_k = find_key(padp_files, match)
        dp_k   = find_key(dp40_files, match)
        padp_val = None
        dp_val = None
        if padp_k:
            padp_val = padp_files[padp_k].get('per_pos_mse')
            padp_val = padp_val[0] if padp_val and len(padp_val)>0 else None
        if dp_k:
            dp_val = dp40_files[dp_k].get('per_pos_mse')
            dp_val = dp_val[0] if dp_val and len(dp_val)>0 else None
        padp_mse0.append(padp_val if padp_val is not None else float('nan'))
        dp40_mse0.append(dp_val if dp_val is not None else float('nan'))
        padp_succ.append(PADP_success[i])
        dp40_succ.append(DP_success[i])
        rows.append((keyname, padp_k, padp_val, dp_k, dp_val, PADP_success[i], DP_success[i]))

    # print table
    # Pretty table
    print(f"{'TASK':20s} | {'PADP_mse0':>10s} | {'DP40_mse0':>10s} | {'PADP_succ':>8s} | {'DP_succ':>7s}")
    print('-'*66)
    for (task, padp_k, padp_v, dp_k, dp_v, padp_s, dp_s) in rows:
        pm = f"{padp_v:.6g}" if padp_v is not None else 'NA'
        dm = f"{dp_v:.6g}" if dp_v is not None else 'NA'
        print(f"{task:20s} | {pm:>10s} | {dm:>10s} | {padp_s:8d} | {dp_s:7d}")

    # correlations
    # use mse0 vs success (lower mse -> higher success => expect negative corr)
    corr_padp_pear = pearson(padp_mse0, padp_succ)
    corr_dp_pear   = pearson(dp40_mse0, dp40_succ)
    corr_padp_spear = spearman(padp_mse0, padp_succ)
    corr_dp_spear   = spearman(dp40_mse0, dp40_succ)

    print('\nPearson correlation (mse0 vs success): PADP={:.4f}, DP40={:.4f}'.format(corr_padp_pear, corr_dp_pear))
    print('Spearman correlation (mse0 vs success): PADP={:.4f}, DP40={:.4f}'.format(corr_padp_spear, corr_dp_spear))

    # Linear regression (OLS) success ~ mse0 for PADP and DP40
    try:
        import numpy as np
        from math import isnan

        def ols(x, y):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            if len(x) < 2:
                return None
            X = np.vstack([np.ones_like(x), x]).T
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X.dot(beta)
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
            return {'intercept': float(beta[0]), 'slope': float(beta[1]), 'r2': float(r2)}

        padp_ols = ols(padp_mse0, padp_succ)
        dp_ols = ols(dp40_mse0, dp40_succ)
        if padp_ols:
            print('\nPADP OLS: success = intercept + slope * mse0')
            print(' intercept={intercept:.4f}, slope={slope:.4f}, R2={r2:.4f}'.format(**padp_ols))
        if dp_ols:
            print('\nDP40 OLS: success = intercept + slope * mse0')
            print(' intercept={intercept:.4f}, slope={slope:.4f}, R2={r2:.4f}'.format(**dp_ols))

        # Delta analysis: method difference per-task
        delta_mse = [d - p for p, d in zip(padp_mse0, dp40_mse0)]
        delta_succ = [d - p for p, d in zip(padp_succ, dp40_succ)]
        delta_corr = pearson(delta_mse, delta_succ)
        print('\nDelta (DP - PADP) correlation: Pearson(delta_mse, delta_succ) = {:.4f}'.format(delta_corr))

        # Scatter plot
        import matplotlib.pyplot as plt
        import numpy as _np
        outdir = Path('/media/disk7t/PADP1018/accurency_test/outputs/analysis')
        outdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(padp_mse0, padp_succ, label='PADP', color='C0')
        ax.scatter(dp40_mse0, dp40_succ, label='DP (H=40)', color='C1')
        # regression lines
        xs = _np.linspace(min(min(padp_mse0), min(dp40_mse0)), max(max(padp_mse0), max(dp40_mse0)), 100)
        if padp_ols:
            ys = padp_ols['intercept'] + padp_ols['slope'] * xs
            ax.plot(xs, ys, color='C0', linestyle='--', linewidth=1)
        if dp_ols:
            ys2 = dp_ols['intercept'] + dp_ols['slope'] * xs
            ax.plot(xs, ys2, color='C1', linestyle='--', linewidth=1)
        ax.set_xlabel('MSE at position 0')
        ax.set_ylabel('Success rate (%)')
        ax.set_title('MSE0 vs Success (per method)')
        ax.legend()
        ax.grid(alpha=0.3)
        figpath = str(outdir / 'mse0_vs_success.png')
        fig.tight_layout()
        fig.savefig(figpath, dpi=200)
        print('Saved scatter plot to', figpath)
    except Exception as e:
        print('Skipping regression/plotting due to error:', e)

    # Simple interpretation hints
    print('\nInterpretation notes:')
    print('- PADP executes only the first predicted action, so PADP_mse0 is the most relevant predictor for success rate under diffusion_policy.')
    print('- DP (H=40) likely uses additional sampling/selection or multi-step corrections; therefore dp_mse0 alone may not predict its success reliably.')
    print('- The delta analysis (DP - PADP) correlation indicates whether a method with lower mse0 tends to have higher success per task.')

if __name__ == '__main__':
    main()
