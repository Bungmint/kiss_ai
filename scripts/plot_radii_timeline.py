#!/usr/bin/env python3
"""Plot radii-sum vs cumulative GPT calls from log + trajectory artifacts.

Primary source:
- `artifacts/job_*/trajectories/*.yaml`
  - Extract evaluator checkpoints from Bash outputs.
  - Map each checkpoint to a cumulative GPT-call x-position using trajectory
    step counts.

Fallback source:
- `exp/.../log.jsonl`
  - Used when trajectory extraction misses points or trajectories are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_PATTERN = re.compile(r"^\s*BENCHMARK\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
NUM_PATTERN = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RADIISUM_PATTERNS = [
    re.compile(rf'"radii_sum"\s*:\s*{NUM_PATTERN}'),
    re.compile(rf"radii_sum\s*[=:]\s*[~≈]?\s*{NUM_PATTERN}"),
    re.compile(rf"radii_sum\s*≈\s*{NUM_PATTERN}"),
]
COMBINED_PATTERNS = [
    re.compile(rf'"combined_score"\s*:\s*{NUM_PATTERN}'),
    re.compile(rf"combined_score\s*[=:]\s*[~≈]?\s*{NUM_PATTERN}"),
]
FLOAT_ONLY_PATTERN = re.compile(rf"^\s*{NUM_PATTERN}\s*$")
GPT_CALLS_PATTERN = re.compile(r"(?<![_A-Za-z])gpt_calls\s*=\s*(\d+)")
STEP_PATTERN = re.compile(r"Steps:\s*(\d+)\s*/\s*(\d+)")
RUN_START_PATTERN = re.compile(r'"run_start_timestamp"\s*:\s*(\d+)')
RUN_END_PATTERN = re.compile(r'"run_end_timestamp"\s*:\s*(\d+)')
STEP_COUNT_PATTERN = re.compile(r'"step_count"\s*:\s*(\d+)')

SESSION_FILL = {
    "coder": "#dbeafe",
    "monitor": "#ffedd5",
    "pre": "#f3f4f6",
}

ACTION_DOT = {
    "eval": "#2563eb",
    "optimize": "#dc2626",
    "run": "#6b7280",
    "log": "#059669",
    "other": "#111827",
}


@dataclass
class LogRecord:
    idx: int
    timestamp: str
    agent: str
    action: str
    eval_score: float | None
    summary: str


@dataclass
class TrajectoryMeta:
    path: Path
    agent: str
    run_start_ts: int
    run_end_ts: int
    step_count: int


@dataclass
class LocalEvalPoint:
    local_step: int
    radii_sum: float
    inferred: bool


@dataclass
class SessionSpan:
    session_id: int
    agent: str
    start_x: float
    end_x: float


@dataclass
class Point:
    x: float
    radii_sum: float
    action: str
    agent: str
    source: str
    inferred: bool
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create radii-sum timeline vs cumulative GPT calls (SVG)."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to exp/.../log.jsonl. Defaults to latest exp/*/log.jsonl",
    )
    parser.add_argument(
        "--artifacts-job",
        type=Path,
        default=None,
        help="Path to artifacts/job_* folder. Defaults to latest artifacts/job_*",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to <run_dir>/radii_timeline.svg",
    )
    parser.add_argument(
        "--benchmark",
        type=float,
        default=None,
        help="Benchmark constant for score->radii conversion",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=None,
        help="Path to evaluator.py (auto-read BENCHMARK)",
    )
    parser.add_argument(
        "--include-log-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log.jsonl points when trajectory extraction misses checkpoints",
    )
    parser.add_argument(
        "--strict-trajectories",
        action="store_true",
        help="Fail if no trajectory checkpoints are found",
    )
    parser.add_argument(
        "--shade-sessions",
        action="store_true",
        help="Shade coder/monitor trajectory sessions",
    )
    parser.add_argument(
        "--shade-alpha",
        type=float,
        default=0.12,
        help="Background session shade opacity in [0,1]",
    )
    parser.add_argument("--width", type=int, default=1280, help="SVG width")
    parser.add_argument("--height", type=int, default=760, help="SVG height")
    return parser.parse_args()


def find_latest_log() -> Path:
    candidates = sorted(Path("exp").glob("*/log.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No exp/*/log.jsonl files found")
    return candidates[0]


def find_latest_artifacts_job() -> Path:
    candidates = sorted(Path("artifacts").glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No artifacts/job_* folders found")
    return candidates[0]


def parse_benchmark_from_evaluator(evaluator_path: Path) -> float | None:
    if not evaluator_path.exists():
        return None
    for line in evaluator_path.read_text(encoding="utf-8").splitlines():
        match = BENCHMARK_PATTERN.match(line)
        if match:
            return float(match.group(1))
    return None


def _deep_find_numeric(value: Any, target_key: str) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == target_key and isinstance(item, (int, float)):
                return float(item)
            found = _deep_find_numeric(item, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _deep_find_numeric(item, target_key)
            if found is not None:
                return found
    return None


def parse_radii_from_text(text: str) -> float | None:
    for pattern in RADIISUM_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def parse_combined_from_text(text: str) -> float | None:
    for pattern in COMBINED_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def parse_radii_from_payload(payload: str, benchmark: float | None) -> tuple[float, bool] | None:
    payload = payload.strip()
    if not payload:
        return None

    # Plain score output from evaluator.py
    score_match = FLOAT_ONLY_PATTERN.match(payload)
    if score_match and benchmark is not None:
        return float(score_match.group(1)) * benchmark, True

    # JSON-like output
    if payload.startswith("{") and payload.endswith("}"):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            rs = _deep_find_numeric(obj, "radii_sum")
            if rs is not None:
                return rs, False
            cs = _deep_find_numeric(obj, "combined_score")
            if cs is not None and benchmark is not None:
                return cs * benchmark, True

    # Text fallback
    rs = parse_radii_from_text(payload)
    if rs is not None:
        return rs, False
    cs = parse_combined_from_text(payload)
    if cs is not None and benchmark is not None:
        return cs * benchmark, True

    return None


def parse_log(log_path: Path) -> list[LogRecord]:
    rows: list[LogRecord] = []
    with log_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = obj.get("eval_score")
            rows.append(
                LogRecord(
                    idx=i,
                    timestamp=str(obj.get("timestamp", "")),
                    agent=str(obj.get("agent", "")),
                    action=str(obj.get("action", "")),
                    eval_score=float(score) if isinstance(score, (int, float)) else None,
                    summary=str(obj.get("summary", "")),
                )
            )
    return rows


def parse_gpt_calls(summary: str) -> int | None:
    match = GPT_CALLS_PATTERN.search(summary)
    if not match:
        return None
    return int(match.group(1))


def infer_agent_from_trajectory_name(path: Path) -> str:
    name = path.name.lower()
    if "trajectory_coder_" in name:
        return "coder"
    if "trajectory_monitor_" in name:
        return "monitor"
    return "other"


def parse_trajectory_meta(path: Path) -> TrajectoryMeta | None:
    text = path.read_text(encoding="utf-8")
    m_start = RUN_START_PATTERN.search(text)
    m_end = RUN_END_PATTERN.search(text)
    m_count = STEP_COUNT_PATTERN.search(text)
    if not (m_start and m_end and m_count):
        return None
    return TrajectoryMeta(
        path=path,
        agent=infer_agent_from_trajectory_name(path),
        run_start_ts=int(m_start.group(1)),
        run_end_ts=int(m_end.group(1)),
        step_count=int(m_count.group(1)),
    )


def extract_eval_points_from_trajectory(path: Path, benchmark: float | None) -> list[LocalEvalPoint]:
    """Extract evaluator checkpoints from one trajectory text.

    Logic:
    - detect evaluator-related Bash command
    - read "Steps: k/..." from the same model block
    - parse subsequent `[Bash]: ...` output for radii/score
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    points: list[LocalEvalPoint] = []

    pending_eval = False
    pending_step: int | None = None
    last_step: int | None = None
    capturing_json = False
    json_lines: list[str] = []

    for line in lines:
        if capturing_json:
            stripped = line.strip()
            json_lines.append(stripped)
            if stripped == "}":
                payload = "\n".join(json_lines)
                parsed = parse_radii_from_payload(payload, benchmark)
                if parsed is not None:
                    step = pending_step if pending_step is not None else last_step
                    if step is not None:
                        points.append(
                            LocalEvalPoint(local_step=step, radii_sum=parsed[0], inferred=parsed[1])
                        )
                capturing_json = False
                json_lines = []
                pending_eval = False
                pending_step = None
            continue

        if "Bash(command=" in line and (
            "evaluator.py" in line or "ev.evaluate(" in line or "evaluate('initial_program.py')" in line
        ):
            pending_eval = True
            pending_step = None

        step_match = STEP_PATTERN.search(line)
        if step_match:
            last_step = int(step_match.group(1))
            if pending_eval and pending_step is None:
                pending_step = last_step

        if pending_eval:
            idx = line.find("[Bash]:")
            if idx == -1:
                continue
            payload = line[idx + len("[Bash]:") :].strip()
            if payload.startswith("Error"):
                pending_eval = False
                pending_step = None
                continue
            if payload.startswith("{") and not payload.endswith("}"):
                capturing_json = True
                json_lines = [payload]
                continue

            parsed = parse_radii_from_payload(payload, benchmark)
            if parsed is not None:
                step = pending_step if pending_step is not None else last_step
                if step is not None:
                    points.append(
                        LocalEvalPoint(local_step=step, radii_sum=parsed[0], inferred=parsed[1])
                    )
            pending_eval = False
            pending_step = None

    return points


def collect_trajectories(artifacts_job: Path) -> list[TrajectoryMeta]:
    tdir = artifacts_job / "trajectories"
    if not tdir.exists():
        return []
    metas: list[TrajectoryMeta] = []
    for path in tdir.glob("trajectory_*.yaml"):
        meta = parse_trajectory_meta(path)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: (m.run_start_ts, m.run_end_ts, m.path.name))
    return metas


def build_points_from_trajectories(
    metas: list[TrajectoryMeta], benchmark: float | None
) -> tuple[list[Point], list[SessionSpan], int]:
    points: list[Point] = []
    spans: list[SessionSpan] = []
    cumulative_calls = 0

    for sid, meta in enumerate(metas):
        step_count = max(meta.step_count, 0)
        if step_count > 0:
            spans.append(
                SessionSpan(
                    session_id=sid,
                    agent=meta.agent,
                    start_x=cumulative_calls + 0.5,
                    end_x=cumulative_calls + step_count + 0.5,
                )
            )

        local_points = extract_eval_points_from_trajectory(meta.path, benchmark)
        for local in local_points:
            if step_count <= 0:
                continue
            step = min(max(local.local_step, 1), step_count)
            points.append(
                Point(
                    x=float(cumulative_calls + step),
                    radii_sum=local.radii_sum,
                    action="eval",
                    agent=meta.agent,
                    source="trajectory",
                    inferred=local.inferred,
                    timestamp=str(meta.run_end_ts),
                )
            )

        cumulative_calls += step_count

    points.sort(key=lambda p: p.x)
    return points, spans, cumulative_calls


def build_points_from_log(records: list[LogRecord], benchmark: float | None) -> tuple[list[Point], int]:
    points: list[Point] = []
    cumulative_calls = 0
    for rec in records:
        gpt_calls = parse_gpt_calls(rec.summary)
        if gpt_calls is not None:
            cumulative_calls += gpt_calls

        # Keep eval/optimize checkpoints only for fallback timeline.
        if rec.action not in {"eval", "optimize"}:
            continue

        radii_sum = parse_radii_from_text(rec.summary)
        inferred = False
        if radii_sum is None and rec.eval_score is not None and benchmark is not None:
            radii_sum = rec.eval_score * benchmark
            inferred = True
        if radii_sum is None:
            continue

        points.append(
            Point(
                x=float(cumulative_calls),
                radii_sum=float(radii_sum),
                action=rec.action,
                agent=rec.agent,
                source="log",
                inferred=inferred,
                timestamp=rec.timestamp,
            )
        )
    return points, cumulative_calls


def build_log_session_spans(records: list[LogRecord]) -> list[SessionSpan]:
    spans: list[SessionSpan] = []
    current_agent = "pre"
    start = 0.5
    sid = 0
    cumulative_calls = 0

    for rec in records:
        if rec.action == "run" and rec.agent in {"coder", "monitor"}:
            end = cumulative_calls + 0.5
            if end >= start:
                spans.append(SessionSpan(session_id=sid, agent=current_agent, start_x=start, end_x=end))
            sid += 1
            current_agent = rec.agent
            start = cumulative_calls + 0.5

        gpt_calls = parse_gpt_calls(rec.summary)
        if gpt_calls is not None:
            cumulative_calls += gpt_calls

    end = cumulative_calls + 0.5
    if end >= start:
        spans.append(SessionSpan(session_id=sid, agent=current_agent, start_x=start, end_x=end))
    return spans


def merge_points_with_log_fallback(traj_points: list[Point], log_points: list[Point]) -> list[Point]:
    if not traj_points:
        return sorted(log_points, key=lambda p: p.x)

    existing_calls = {int(round(p.x)) for p in traj_points}
    merged = list(traj_points)
    for p in log_points:
        if int(round(p.x)) in existing_calls:
            continue
        merged.append(p)
    merged.sort(key=lambda p: p.x)
    return merged


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def make_svg(
    *,
    points: list[Point],
    spans: list[SessionSpan],
    title: str,
    subtitle: str,
    width: int,
    height: int,
    shade_sessions: bool,
    shade_alpha: float,
) -> str:
    margin_left = 96
    margin_right = 30
    margin_top = 84
    margin_bottom = 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min = min(p.x for p in points)
    x_max = max(p.x for p in points)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    y_min = min(p.radii_sum for p in points)
    y_max = max(p.radii_sum for p in points)
    if math.isclose(y_min, y_max):
        pad = max(abs(y_max) * 0.02, 0.01)
    else:
        pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return margin_top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    def ticks(low: float, high: float, n: int) -> list[float]:
        if n <= 1:
            return [low]
        step = (high - low) / (n - 1)
        return [low + i * step for i in range(n)]

    y_ticks = ticks(y_min, y_max, n=6)
    x_ticks = ticks(x_min, x_max, n=9)

    points_sorted = sorted(points, key=lambda p: p.x)
    polyline = " ".join(f"{sx(p.x):.2f},{sy(p.radii_sum):.2f}" for p in points_sorted)

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 {w} {h}" font-family="ui-sans-serif, -apple-system, Segoe UI, Arial">'.format(
            w=width, h=height
        )
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    if shade_sessions:
        alpha = min(max(shade_alpha, 0.0), 1.0)
        for span in spans:
            left = sx(max(span.start_x, x_min))
            right = sx(min(span.end_x, x_max))
            if right <= left:
                continue
            fill = SESSION_FILL.get(span.agent, SESSION_FILL["pre"])
            lines.append(
                f'<rect x="{left:.2f}" y="{margin_top}" width="{right - left:.2f}" height="{plot_h}" '
                f'fill="{fill}" fill-opacity="{alpha:.3f}"/>'
            )

    lines.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#d1d5db" stroke-width="1"/>'
    )

    for y in y_ticks:
        yp = sy(y)
        lines.append(
            f'<line x1="{margin_left}" y1="{yp:.2f}" x2="{margin_left + plot_w}" y2="{yp:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{yp + 4:.2f}" text-anchor="end" font-size="12" fill="#374151">'
            f"{y:.6f}</text>"
        )

    for x in x_ticks:
        xp = sx(x)
        lines.append(
            f'<line x1="{xp:.2f}" y1="{margin_top}" x2="{xp:.2f}" y2="{margin_top + plot_h}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xp:.2f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#374151">'
            f"{int(round(x))}</text>"
        )

    lines.append(
        f'<polyline points="{polyline}" fill="none" stroke="#111827" stroke-width="2.25" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for p in points_sorted:
        dot_color = ACTION_DOT.get(p.action, ACTION_DOT["other"])
        stroke = "#ffffff" if p.source == "trajectory" else ACTION_DOT["log"]
        stroke_w = "1.2" if p.source == "trajectory" else "1.9"
        lines.append(
            f'<circle cx="{sx(p.x):.2f}" cy="{sy(p.radii_sum):.2f}" r="4.3" fill="{dot_color}" '
            f'stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )

    lines.append(
        f'<text x="{margin_left}" y="34" font-size="23" font-weight="700" fill="#111827">{escape_xml(title)}</text>'
    )
    lines.append(
        f'<text x="{margin_left}" y="57" font-size="13" fill="#4b5563">{escape_xml(subtitle)}</text>'
    )

    lines.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 26}" text-anchor="middle" '
        'font-size="13" fill="#111827">Cumulative GPT Calls</text>'
    )
    lines.append(
        f'<text x="24" y="{margin_top + plot_h / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {margin_top + plot_h / 2:.2f})" font-size="13" fill="#111827">'
        "Sum of Radii</text>"
    )

    legend_x = margin_left + 2
    legend_y = margin_top + 10
    legend_h = 118 if shade_sessions else 92
    lines.append(
        f'<rect x="{legend_x - 3}" y="{legend_y - 18}" width="390" height="{legend_h}" '
        'fill="#ffffff" fill-opacity="0.9" stroke="#e5e7eb" rx="6"/>'
    )
    lines.append(
        f'<line x1="{legend_x + 8}" y1="{legend_y}" x2="{legend_x + 44}" y2="{legend_y}" '
        'stroke="#111827" stroke-width="2.25"/>'
    )
    lines.append(
        f'<circle cx="{legend_x + 26}" cy="{legend_y}" r="4.3" fill="{ACTION_DOT["eval"]}" stroke="#fff" stroke-width="1.2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 54}" y="{legend_y + 4}" font-size="12.5" fill="#111827">Radii timeline</text>'
    )

    lines.append(
        f'<circle cx="{legend_x + 26}" cy="{legend_y + 24}" r="4.3" fill="{ACTION_DOT["eval"]}" '
        f'stroke="#fff" stroke-width="1.2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 54}" y="{legend_y + 28}" font-size="12.5" fill="#111827">Trajectory checkpoint</text>'
    )

    lines.append(
        f'<circle cx="{legend_x + 26}" cy="{legend_y + 46}" r="4.3" fill="{ACTION_DOT["optimize"]}" '
        f'stroke="{ACTION_DOT["log"]}" stroke-width="1.9"/>'
    )
    lines.append(
        f'<text x="{legend_x + 54}" y="{legend_y + 50}" font-size="12.5" fill="#111827">Log fallback point</text>'
    )

    if shade_sessions:
        sw_x = legend_x + 210
        alpha = min(max(shade_alpha, 0.0), 1.0)
        lines.append(
            f'<rect x="{sw_x}" y="{legend_y + 14}" width="14" height="10" fill="{SESSION_FILL["coder"]}" fill-opacity="{alpha:.3f}"/>'
        )
        lines.append(
            f'<text x="{sw_x + 20}" y="{legend_y + 23}" font-size="12.5" fill="#111827">Coder session</text>'
        )
        lines.append(
            f'<rect x="{sw_x}" y="{legend_y + 36}" width="14" height="10" fill="{SESSION_FILL["monitor"]}" fill-opacity="{alpha:.3f}"/>'
        )
        lines.append(
            f'<text x="{sw_x + 20}" y="{legend_y + 45}" font-size="12.5" fill="#111827">Monitor session</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    log_path = args.log if args.log is not None else find_latest_log()
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    records = parse_log(log_path)
    if not records:
        raise ValueError(f"No records in log: {log_path}")

    artifacts_job = args.artifacts_job if args.artifacts_job is not None else find_latest_artifacts_job()
    if not artifacts_job.exists():
        raise FileNotFoundError(f"artifacts job not found: {artifacts_job}")

    benchmark = args.benchmark
    if benchmark is None:
        evaluator_path = args.evaluator if args.evaluator is not None else (log_path.parent / "repo" / "evaluator.py")
        benchmark = parse_benchmark_from_evaluator(evaluator_path)

    trajectory_metas = collect_trajectories(artifacts_job)
    trajectory_points, trajectory_spans, trajectory_total_calls = build_points_from_trajectories(
        trajectory_metas, benchmark
    )

    log_points, log_total_calls = build_points_from_log(records, benchmark)

    if args.strict_trajectories and not trajectory_points:
        raise ValueError(f"No trajectory checkpoints found in: {artifacts_job / 'trajectories'}")

    if trajectory_points:
        points = (
            merge_points_with_log_fallback(trajectory_points, log_points)
            if args.include_log_fallback
            else trajectory_points
        )
        spans = trajectory_spans
        x_total_calls = trajectory_total_calls
    else:
        points = log_points
        spans = build_log_session_spans(records)
        x_total_calls = log_total_calls

    if not points:
        hint = ""
        if benchmark is None:
            hint = " (benchmark missing, so score-only outputs cannot be converted to radii_sum)"
        raise ValueError(f"No plottable radii points found{hint}.")

    output_path = args.output if args.output is not None else (log_path.parent / "radii_timeline.svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inferred_count = sum(1 for p in points if p.inferred)
    traj_count = sum(1 for p in points if p.source == "trajectory")
    log_count = sum(1 for p in points if p.source == "log")

    title = f"Radii Sum vs GPT Calls: {log_path.parent.name}"
    subtitle = (
        f"{len(points)} points ({traj_count} trajectory, {log_count} log fallback), "
        f"total_calls≈{x_total_calls}, benchmark={benchmark:.10g}" if benchmark is not None else
        f"{len(points)} points ({traj_count} trajectory, {log_count} log fallback), total_calls≈{x_total_calls}"
    )
    if inferred_count > 0:
        subtitle += f", inferred={inferred_count}"

    svg = make_svg(
        points=points,
        spans=spans,
        title=title,
        subtitle=subtitle,
        width=args.width,
        height=args.height,
        shade_sessions=args.shade_sessions,
        shade_alpha=args.shade_alpha,
    )
    output_path.write_text(svg, encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Log: {log_path}")
    print(f"Artifacts job: {artifacts_job}")
    print(f"Trajectory files: {len(trajectory_metas)}")
    print(f"Trajectory checkpoints: {sum(1 for p in trajectory_points)}")
    print(f"Log fallback checkpoints: {sum(1 for p in points if p.source == 'log')}")
    print(f"Plotted points: {len(points)}")
    print(f"Estimated total GPT calls: {x_total_calls}")
    if benchmark is not None:
        print(f"Benchmark: {benchmark}")
    if inferred_count:
        print(f"Inferred radii_sum count: {inferred_count}")
    if trajectory_total_calls and log_total_calls and trajectory_total_calls != log_total_calls:
        print(
            "Warning: trajectory total calls and log total calls differ "
            f"({trajectory_total_calls} vs {log_total_calls})."
        )


if __name__ == "__main__":
    main()
