#!/usr/bin/env python3
"""Compare three present-moment clarity curves for study E1.

Hyperbola V = C / (C + N), exponential V = exp(-N / C), and a line
clipped to 0 and 1. Parameters are fitted only on the training
participants. The held-out half is not used in the fit.

The demonstration run is not project data and does not confirm the
hypothesis. A real file of sessions has not been collected.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


GRID = [i / 20 for i in range(1, 4001)]  # C from 0.05 to 200
MARGIN = 0.005  # RMSE gap below which H3 in the E1 plan is not supported
STEPS = [0, 1, 2, 4]


def to_unit(values: list[float]) -> list[float]:
    if values and max(values) > 1:
        return [v / 100 for v in values]
    return values


def rmse(pred: list[float], obs: list[float]) -> float:
    if not obs:
        return math.nan
    return math.sqrt(sum((p - o) ** 2 for p, o in zip(pred, obs)) / len(obs))


def hyperbola(n: float, c: float) -> float:
    return c / (c + n)


def exponential(n: float, c: float) -> float:
    return math.exp(-n / c)


def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def fit_c(ns: list[float], vs: list[float], fn) -> float:
    best_c = GRID[0]
    best_sse = math.inf
    for c in GRID:
        sse = sum((fn(n, c) - v) ** 2 for n, v in zip(ns, vs))
        if sse < best_sse:
            best_sse = sse
            best_c = c
    return best_c


def fit_line(ns: list[float], vs: list[float]) -> tuple[float, float]:
    n = len(ns)
    if n == 0:
        return 1.0, 0.0
    mean_n = sum(ns) / n
    mean_v = sum(vs) / n
    var_n = sum((x - mean_n) ** 2 for x in ns)
    if var_n == 0:
        return mean_v, 0.0
    cov = sum((x - mean_n) * (y - mean_v) for x, y in zip(ns, vs))
    slope = cov / var_n
    intercept = mean_v - slope * mean_n
    return intercept, slope


def predict(ns: list[float], kind: str, params) -> list[float]:
    if kind == "hyperbola":
        return [hyperbola(n, params) for n in ns]
    if kind == "exponential":
        return [exponential(n, params) for n in ns]
    intercept, slope = params
    return [clip01(intercept + slope * n) for n in ns]


def split_ids(ids: list[str]) -> tuple[set[str], set[str]]:
    """Positions from one. Odd positions are the test set, even positions the training set. E1 plan."""
    ordered = sorted(set(ids))
    train = {pid for i, pid in enumerate(ordered, start=1) if i % 2 == 0}
    test = {pid for i, pid in enumerate(ordered, start=1) if i % 2 == 1}
    return train, test


def ready(points: list[dict]) -> list[dict]:
    """Keep only people with one row for each step 0, 1, 2, 4."""
    counts: dict[str, list[float]] = {}
    for point in points:
        counts.setdefault(point["id"], []).append(point["n"])
    keep = {pid for pid, ns in counts.items() if sorted(ns) == STEPS}
    return [point for point in points if point["id"] in keep]


def n_half(steps: list[tuple[float, float]]) -> float | None:
    """N at which clarity falls to half the clarity at the smallest N."""
    ordered = sorted(steps, key=lambda item: item[0])
    if len(ordered) < 2:
        return None
    baseline = ordered[0][1]
    target = baseline / 2
    for (n0, v0), (n1, v1) in zip(ordered, ordered[1:]):
        if v0 >= target >= v1 and v0 != v1:
            share = (v0 - target) / (v0 - v1)
            return n0 + share * (n1 - n0)
        if v0 >= target >= v1 and v0 == v1:
            return n1
    return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_to_points(rows: list[dict[str, str]]) -> list[dict]:
    points = []
    raw_v = []
    parsed = []
    for row in rows:
        if row.get("included", "1") in {"0", "false", "False"}:
            continue
        n = float(row["n"])
        v = float(row["v"])
        parsed.append((row["participant_id"], n, v, row))
        raw_v.append(v)
    unit = to_unit(raw_v)
    for (pid, n, _v, row), v in zip(parsed, unit):
        points.append({"id": pid, "n": n, "v": v, "row": row})
    return points


def compare(points: list[dict]) -> dict:
    train_ids, test_ids = split_ids([p["id"] for p in points])
    train = [p for p in points if p["id"] in train_ids]
    test = [p for p in points if p["id"] in test_ids]
    if not train or not test:
        raise SystemExit("Both training and held-out participants are required. One side of the ordered id split is empty.")
    ns = [p["n"] for p in train]
    vs = [p["v"] for p in train]
    c_h = fit_c(ns, vs, hyperbola)
    c_e = fit_c(ns, vs, exponential)
    line = fit_line(ns, vs)
    test_n = [p["n"] for p in test]
    test_v = [p["v"] for p in test]
    scores = {
        "hyperbola": rmse(predict(test_n, "hyperbola", c_h), test_v),
        "exponential": rmse(predict(test_n, "exponential", c_e), test_v),
        "line": rmse(predict(test_n, "line", line), test_v),
    }
    winner = min(scores, key=scores.get)
    return {
        "train_people": len(train_ids),
        "test_people": len(test_ids),
        "c_hyperbola": c_h,
        "c_exponential": c_e,
        "line": line,
        "rmse": scores,
        "winner": winner,
        "h3_alive": scores["exponential"] - scores["hyperbola"] >= MARGIN
        and scores["line"] - scores["hyperbola"] >= MARGIN,
    }


def silence_check(points: list[dict]) -> str:
    """H18: clarity of a valid quiet probe predicts the half-clarity N in held-out people."""
    by_person: dict[str, list[tuple[float, float]]] = {}
    silence: dict[str, float] = {}
    for point in points:
        by_person.setdefault(point["id"], []).append((point["n"], point["v"]))
        raw = point["row"].get("v_empty", "")
        valid = point["row"].get("empty_valid", "")
        if raw and valid in {"1", "true", "True"}:
            value = float(raw)
            silence[point["id"]] = value / 100 if value > 1 else value
    _train, test_ids = split_ids(list(by_person))
    pairs = []
    for pid in sorted(test_ids):
        if pid not in silence:
            continue
        half = n_half(by_person[pid])
        if half is not None:
            pairs.append((silence[pid], half))
    if len(pairs) < 3:
        return "H18 not computed: fewer than three valid quiet-probe and half-point pairs in the held-out set."
    mean_s = sum(s for s, _h in pairs) / len(pairs)
    mean_h = sum(h for _s, h in pairs) / len(pairs)
    cov = sum((s - mean_s) * (h - mean_h) for s, h in pairs)
    var_s = sum((s - mean_s) ** 2 for s, _h in pairs)
    var_h = sum((h - mean_h) ** 2 for _s, h in pairs)
    if var_s == 0 or var_h == 0:
        return "H18 not computed: the held-out quiet probe or the half-clarity N has no spread."
    corr = cov / math.sqrt(var_s * var_h)
    alive = corr > 0
    return f"H18 on held-out people: correlation of quiet-probe clarity and half-clarity N = {corr:.3f}. Supported by sign: {alive}."


def h30_note(points: list[dict]) -> str:
    """Two mean drops on held-out people. H30 has no verdict in E1."""
    _train, test_ids = split_ids([point["id"] for point in points])
    by: dict[str, dict[float, float]] = {}
    for point in points:
        if point["id"] in test_ids:
            by.setdefault(point["id"], {})[point["n"]] = point["v"]
    first: list[float] = []
    late: list[float] = []
    for row in by.values():
        if all(step in row for step in STEPS):
            first.append(row[0] - row[1])
            late.append(row[2] - row[4])
    if not first:
        return "H30 not computed: held-out people do not have the full set of steps."
    return (
        "H30 description, no verdict: "
        f"mean drop in clarity from 0 to 1 = {sum(first) / len(first):.3f}; "
        f"from 2 to 4 = {sum(late) / len(late):.3f}. People: {len(first)}."
    )


def h50_note(points: list[dict]) -> str:
    """Quiet-probe clarity among those who would stay, against those who would leave."""
    answered = [point for point in points if point["row"].get("stay", "") in {"0", "1"}]
    if not answered:
        return "H50 is absent from the file: the stay column is empty."
    _train, test_ids = split_ids([point["id"] for point in points])
    stay: dict[str, float] = {}
    leave: dict[str, float] = {}
    seen: set[str] = set()
    for point in points:
        pid = point["id"]
        if pid not in test_ids or pid in seen:
            continue
        raw = point["row"].get("v_empty", "")
        valid = point["row"].get("empty_valid", "")
        choice = point["row"].get("stay", "")
        if not raw or valid not in {"1", "true", "True"} or choice not in {"0", "1"}:
            continue
        seen.add(pid)
        value = float(raw)
        value = value / 100 if value > 1 else value
        if choice == "1":
            stay[pid] = value
        else:
            leave[pid] = value
    if not stay or not leave:
        return "H50 is not supported in this file: the held-out valid quiet probes do not contain both answers."
    mean_stay = sum(stay.values()) / len(stay)
    mean_leave = sum(leave.values()) / len(leave)
    alive = mean_stay > mean_leave
    return (
        f"H50: quiet-probe clarity among those who would stay, {mean_stay:.3f}; "
        f"among those who would leave, {mean_leave:.3f}. Supported: {alive}."
    )


def write_demo(path: Path, kind: str, seed: int) -> None:
    rng = random.Random(seed)
    fieldnames = ["participant_id", "n", "v", "v_empty", "empty_valid", "included"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(40):
            c = rng.uniform(1.2, 6.0)
            silence = rng.uniform(40, 95)
            for n in (0, 1, 2, 4):
                if kind == "hyperbola":
                    v = hyperbola(n, c)
                elif kind == "line":
                    v = clip01(0.9 - 0.18 * n)
                else:
                    v = exponential(n, c)
                v = clip01(v + rng.gauss(0, 0.03))
                writer.writerow(
                    {
                        "participant_id": f"demo-{i:02d}",
                        "n": n,
                        "v": round(v * 100, 2),
                        "v_empty": round(silence, 2),
                        "empty_valid": "1",
                        "included": "1",
                    }
                )


def test_points(points: list[dict]) -> list[dict]:
    _train, test_ids = split_ids([point["id"] for point in points])
    return [point for point in points if point["id"] in test_ids]


def samples(result: dict, steps: int = 81) -> tuple[list[float], dict[str, list[float]]]:
    xs = [4 * i / (steps - 1) for i in range(steps)]
    return xs, {
        "hyperbola": predict(xs, "hyperbola", result["c_hyperbola"]),
        "exponential": predict(xs, "exponential", result["c_exponential"]),
        "line": predict(xs, "line", result["line"]),
    }


def write_curves_svg(path: Path, panels: list[tuple[str, list[dict], dict]]) -> None:
    """Three panels: the shape at one C, and two synthetic sets. This is not project data."""
    width, height = 1080, 460
    panel_w, panel_h = 300, 260
    origin_x, origin_y = 56, 84
    gap = 40

    def xy(n: float, v: float, left: float) -> tuple[float, float]:
        x = left + (n / 4) * panel_w
        y = origin_y + (1 - min(1.0, max(0.0, v))) * panel_h
        return x, y

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="32" font-family="sans-serif" font-size="16" fill="#222">E1 synthetic figure. Not data and not confirmation.</text>',
        '<text x="48" y="52" font-family="sans-serif" font-size="12" fill="#555">Points are held-out people. Lines are curves fitted on the other half. Clarity is a share from 0 to 1.</text>',
    ]
    colors = {"hyperbola": "#c2410c", "exponential": "#1d4ed8", "line": "#15803d"}
    names = {"hyperbola": "hyperbola", "exponential": "exponential", "line": "line"}
    for index, (title, points, result) in enumerate(panels):
        left = origin_x + index * (panel_w + gap)
        lines.append(f'<text x="{left}" y="{origin_y - 14}" font-family="sans-serif" font-size="13" fill="#222">{title}</text>')
        lines.append(
            f'<rect x="{left}" y="{origin_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#ddd"/>'
        )
        for tick in (0, 1, 2, 4):
            x, _y = xy(tick, 0, left)
            lines.append(f'<text x="{x - 4}" y="{origin_y + panel_h + 16}" font-family="sans-serif" font-size="11" fill="#555">{tick}</text>')
        for tick, label in ((0, "0"), (0.5, "0.5"), (1, "1")):
            _x, y = xy(0, tick, left)
            lines.append(f'<text x="{left - 28}" y="{y + 4}" font-family="sans-serif" font-size="11" fill="#555">{label}</text>')
        if points:
            for point in points:
                x, y = xy(point["n"], point["v"], left)
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#94a3b8"/>')
        xs, ys = samples(result)
        for kind in ("line", "exponential", "hyperbola"):
            coords = " ".join(f"{xy(n, v, left)[0]:.1f},{xy(n, v, left)[1]:.1f}" for n, v in zip(xs, ys[kind]))
            lines.append(f'<polyline fill="none" stroke="{colors[kind]}" stroke-width="2" points="{coords}"/>')
        verdict = result.get("caption")
        if not verdict:
            verdict = "H3 supported" if result["h3_alive"] else "H3 not supported"
        lines.append(
            f'<text x="{left}" y="{origin_y + panel_h + 36}" font-family="sans-serif" font-size="12" fill="#222">{verdict}</text>'
        )
    for row, kind in enumerate(("hyperbola", "exponential", "line")):
        y = 440
        lines.append(f'<line x1="{48 + row * 140}" y1="{y}" x2="{76 + row * 140}" y2="{y}" stroke="{colors[kind]}" stroke-width="2"/>')
        lines.append(
            f'<text x="{82 + row * 140}" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#222">{names[kind]}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def shape_panel() -> tuple[str, list[dict], dict]:
    """One value C = 2 with no noise: how the three formulas look, not a study result."""
    return (
        "Shape at C = 2, no noise",
        [],
        {
            "c_hyperbola": 2.0,
            "c_exponential": 2.0,
            "line": (1.0, -1 / 6),
            "h3_alive": False,
            "caption": "formula, not a test",
        },
    )


def report(title: str, result: dict, silence: str, price: str, choice: str) -> None:
    print(title)
    print(f"  training people: {result['train_people']}, held out: {result['test_people']}")
    print(f"  hyperbola C: {result['c_hyperbola']:.2f}")
    print(f"  exponential C: {result['c_exponential']:.2f}")
    intercept, slope = result["line"]
    print(f"  line: {intercept:.3f} + ({slope:.3f}) N, then clipped to 0 and 1")
    for name, score in result["rmse"].items():
        print(f"  RMSE {name}: {score:.4f}")
    print(f"  smaller error: {result['winner']}")
    print(f"  H3 supported by the E1 rule: {result['h3_alive']}")
    print(f"  {silence}")
    print(f"  {price}")
    print(f"  {choice}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three clarity curves on a session file.")
    parser.add_argument("csv", nargs="?", help="CSV with columns participant_id, n, v")
    parser.add_argument("--demo", action="store_true", help="Run synthetic series and check that the code does not always crown the hyperbola")
    args = parser.parse_args()
    if args.demo or not args.csv:
        folder = Path(__file__).resolve().parent / "_demo"
        folder.mkdir(exist_ok=True)
        hyp_path = folder / "hyperbola.csv"
        line_path = folder / "line.csv"
        write_demo(hyp_path, "hyperbola", 1)
        write_demo(line_path, "line", 2)
        print("Demonstration. This is not project data and not confirmation of the formula.")
        hyp_points = ready(rows_to_points(load_rows(hyp_path)))
        line_points = ready(rows_to_points(load_rows(line_path)))
        hyp_result = compare(hyp_points)
        line_result = compare(line_points)
        report(
            "Synthetic series from the hyperbola",
            hyp_result,
            silence_check(hyp_points),
            h30_note(hyp_points),
            h50_note(hyp_points),
        )
        report(
            "Synthetic series from the line",
            line_result,
            silence_check(line_points),
            h30_note(line_points),
            h50_note(line_points),
        )
        if not hyp_result["h3_alive"] or line_result["winner"] == "hyperbola":
            raise SystemExit("The code check failed: the hyperbola must win on its own synthetic series and lose on the line.")
        picture = folder / "curves.svg"
        write_curves_svg(
            picture,
            [
                shape_panel(),
                ("Synthetic series from the hyperbola", test_points(hyp_points), hyp_result),
                ("Synthetic series from the line", test_points(line_points), line_result),
            ],
        )
        print("The code check passed. It has no bearing on the project hypotheses.")
        print(f"Figure: {picture}")
        return
    points = ready(rows_to_points(load_rows(Path(args.csv))))
    result = compare(points)
    report(str(args.csv), result, silence_check(points), h30_note(points), h50_note(points))


if __name__ == "__main__":
    main()
