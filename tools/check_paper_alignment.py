"""Read-only check of all eight website result tables against the supplied TeX.

Usage: python tools/check_paper_alignment.py --paper PATH [--extract]
No dependencies outside the Python standard library; never modifies inputs.
"""
import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

LABELS = [
    "main_simulation_comparison", "method_configurations",
    "single_eval_operating_points", "inference_budget_sweep",
    "component_ablation", "beta_h_selection", "real_world_task1", "real_world_task2",
]
LEGACY_LABELS = ["obs_steps_ablation", "batch_size_ablation", "data_size_ablation"]


def plain(cell):
    cell = re.sub(r"\\cite\{[^}]*\}", "", cell)
    cell = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}", r"\1", cell)
    for _ in range(4):
        cell = re.sub(r"\\(?:textbf|underline|emph)\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace(r"\epsilon", "ε").replace("x_0", "x₀")
    cell = cell.replace(r"\times", "×").replace(r"\%", "%")
    cell = cell.replace("$", "").replace("~", " ")
    return " ".join(cell.split())


def extract(tex):
    tex = re.sub(r"(?<!\\)%[^\n]*", "", tex)
    tables = {}
    for block in re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        label = re.search(r"\\label\{tab:([^}]+)\}", block)
        if not label or label[1] not in LABELS:
            continue
        key = label[1]
        body = re.search(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", block, re.S)[1]
        rows, previous, ta = [], None, None
        for fragment in re.split(r"\\\\", body):
            if "T_a=1" in fragment:
                ta = "1"
            elif "T_a=8" in fragment:
                ta = "8"
            fragment = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", fragment)
            fragment = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", fragment)
            fragment = re.sub(r"\\addlinespace(?:\[[^]]*\])?", "", fragment)
            # Group labels may share a fragment with the following data row.
            fragment = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{(?:[^{}]|\{[^{}]*\})*\}\s*", "", fragment)
            cells = [plain(c) for c in fragment.strip().split("&")]
            if key == "beta_h_selection":
                if len(cells) == 6 and re.fullmatch(r"0(?:\.\d+)?", cells[0]):
                    rows.append(cells)
                elif len(cells) == 5 and all(re.fullmatch(r"\d+", c) for c in cells[1:]):
                    rows.append(["Steady-state latency (ms)", ""] + cells[1:])
                continue
            allowed = ("PASDP", "DP", "SDP", "FP", "Fixed", "Uniform", "No buffer")
            if not cells[0].startswith(allowed):
                if not cells[0] and previous and len(cells) == len(previous):
                    for i in range(3 if key == "inference_budget_sweep" else 1):
                        if not cells[i]:
                            cells[i] = previous[i]
                else:
                    continue
            if len(cells) < 4 or not any(re.fullmatch(r"\d+(?:\.\d+)?", c) for c in cells):
                continue
            previous = cells.copy()
            if key == "single_eval_operating_points":
                cells = [ta] + cells
            rows.append(cells)
        tables[key] = rows
    expected_counts = [4, 4, 9, 13, 5, 8, 6, 2]
    for key, count in zip(LABELS, expected_counts):
        if len(tables.get(key, [])) != count:
            raise ValueError(f"Unexpected row count for {key}: {tables.get(key)}")
    return tables


def extract_legacy(tex):
    """Keep the archived appendix separate from the current manuscript tables."""
    tex = re.sub(r"(?<!\\)%[^\n]*", "", tex)
    tables = {}
    for block in re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        label = re.search(r"\\label\{tab:([^}]+)\}", block)
        if not label or label[1] not in LEGACY_LABELS:
            continue
        body = re.search(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", block, re.S)[1]
        rows, previous = [], None
        for fragment in re.split(r"\\\\", body):
            fragment = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", fragment)
            fragment = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", fragment)
            cells = [plain(c).replace(r"\beta=", "β = ").replace(r"\equiv", "≡ ") for c in fragment.strip().split("&")]
            if label[1] == "obs_steps_ablation":
                if not cells[0] and previous and len(cells) == 7:
                    cells[0] = previous[0]
                if len(cells) != 7 or cells[0] not in ("PADP", "DP"):
                    continue
            elif not re.fullmatch(r"\d+", cells[0]):
                continue
            previous = cells.copy()
            rows.append(cells)
        tables[label[1]] = rows
    for label, count in zip(LEGACY_LABELS, [4, 5, 5]):
        if len(tables.get(label, [])) != count:
            raise ValueError(f"Unexpected legacy row count for {label}: {tables.get(label)}")
    return tables


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = {}, None, None, None
        self.data_rows, self.videos, self.references, self.ids, self.text = False, [], [], [], []
        self.ablation_bars = []
        self.details = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "details" and "id" in attrs:
            self.details[attrs["id"]] = attrs
        if tag == "table":
            self.table = attrs.get("data-paper-table") or attrs.get("data-legacy-table")
            if self.table:
                self.tables[self.table] = []
        if self.table and tag in ("tbody", "tfoot"):
            self.data_rows = True
        if self.table and self.data_rows and tag == "tr":
            self.row = []
        if self.row is not None and tag in ("td", "th"):
            self.cell = []
            self.cell_span = int(attrs.get("colspan", "1"))
        if tag == "video":
            self.videos.append(attrs)
        if tag == "li" and "data-ablation-variant" in attrs:
            self.ablation_bars.append([attrs["data-ablation-variant"], attrs.get("data-success"), attrs.get("data-nfe")])
        for field in ("src", "poster", "href"):
            if field in attrs:
                self.references.append(attrs[field])

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.row.extend([""] * (self.cell_span - 1))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.tables[self.table].append(self.row)
            self.row = None
        if tag in ("tbody", "tfoot"):
            self.data_rows = False
        if tag == "table":
            self.table = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True, type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--legacy-paper", type=Path, help="Optionally compare archived appendix tables directly with their original TRO TeX")
    args = parser.parse_args()
    tex = args.paper.read_text(encoding="utf-8-sig")
    expected = extract(tex)
    if args.extract:
        print(json.dumps(extract_legacy(args.legacy_paper.read_text(encoding="utf-8-sig")) if args.legacy_paper else expected, ensure_ascii=False))
        return
    root = Path(__file__).resolve().parents[1]
    page = PageParser()
    page.feed((root / "index.html").read_text(encoding="utf-8-sig"))
    errors = []
    for label in LABELS:
        if page.tables.get(label) != expected[label]:
            errors.append(f"Table {label} mismatch:\nexpected={expected[label]}\nactual={page.tables.get(label)}")
    if page.ablation_bars != [[row[0], row[12], row[1]] for row in expected["component_ablation"]]:
        errors.append("Component-ablation chart differs from Table V")
    if len(page.videos) != 20 or any(not all(a in v for a in ("autoplay", "loop", "muted", "playsinline")) for v in page.videos):
        errors.append("All 20 videos must retain autoplay / loop / muted / playsinline")
    if "paper" in page.ids or "citation" in page.ids:
        errors.append("Paper and BibTeX must remain hidden")
    if len(page.ids) != len(set(page.ids)):
        errors.append("Duplicate HTML IDs")
    from urllib.parse import unquote, urlsplit
    for ref in page.references:
        parts = urlsplit(ref)
        if not parts.scheme and not parts.netloc and parts.path and not (root / unquote(parts.path)).exists():
            errors.append(f"Missing local asset: {ref}")
    text = " ".join(page.text)
    for stale in ("261 ms", "75.9% with", "7×", "4×+", "one order of magnitude", "best uninterrupted sequence", "Acknowledgements to be added"):
        if stale in text:
            errors.append(f"Stale visible claim: {stale}")
    source_hash = hashlib.sha256(args.paper.read_bytes()).hexdigest()
    provenance = json.loads((root / "assets/data/paper-provenance.json").read_text(encoding="utf-8"))
    if provenance["paper_sha256"] != source_hash:
        errors.append("Paper has changed: update provenance and re-audit narrative / figures")
    if provenance["tables"] != expected:
        errors.append("Public provenance dataset differs from the manuscript")
    legacy = json.loads((root / "assets/data/legacy-development-results.json").read_text(encoding="utf-8"))
    for label in LEGACY_LABELS:
        if page.tables.get(label) != legacy["tables"][label]:
            errors.append(f"Archived development table {label} differs from its recorded source")
    folded = page.details.get("earlier-development-ablations")
    if folded is None or "open" in folded:
        errors.append("Archived development studies must exist and be collapsed by default")
    if args.legacy_paper:
        if hashlib.sha256(args.legacy_paper.read_bytes()).hexdigest() != legacy["source_sha256"]:
            errors.append("Archived TRO source hash changed")
        if extract_legacy(args.legacy_paper.read_text(encoding="utf-8-sig")) != legacy["tables"]:
            errors.append("Archived development snapshot differs from original TRO appendix")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: all 8 tables / {sum(map(len, expected.values()))} rows match {args.paper.name}")
    print("PASS: all local media paths, hidden Paper/BibTeX, 20 muted autoplay videos, source hash")
    print("PASS: 3 archived development tables / 14 rows, kept separate and collapsed by default" + ("; original TRO TeX checked" if args.legacy_paper else "; recorded snapshot checked"))
    print("Scope: manuscript alignment only; no raw experimental evidence is independently verified.")


if __name__ == "__main__":
    main()
