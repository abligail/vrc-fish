#!/usr/bin/env python3
"""
从 `vrc-fish.cpp` 的自动钓鱼 debug 日志里拟合滑块物理参数 (gravity / thrust / drag)。

MPC 里使用的离散模型（每步）:
  v = v * drag + accel
  y = y + v
其中:
  accel = thrust (hold=1) 或 gravity (hold=0)

日志里我们能拿到:
  - [ctrl] ... sCY=... sH=... [color|tpl]   （滑块中心位置）
  - [MPC] ... hold=0/1 dt=...ms            （偶尔打印，用作 hold 标签与 dt 统计）

注意:
  - 这份日志的 [MPC] 行默认每 ~200ms 才打印一次，因此 hold 标签是“分段常量”的近似；
    数据越长越容易靠鲁棒/截尾拟合压住这类噪声。
  - 脚本只依赖 Python 标准库，方便在没有 numpy/scipy 的环境里直接跑。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


CTRL_RE = re.compile(
    r"\[ctrl\] (?P<detect_ms>\d+)ms \[(?P<mode>fast|full)\]"
    r"(?: dt=(?P<dt_ms>\d+)ms)?"
    r"(?: t=(?P<t_ms>\d+))?"
    r" "
    r"fishY=(?P<fishY>\d+) sCY=(?P<sCY>\d+) sH=(?P<sH>\d+) \[(?P<src>color|tpl)\]"
    r"(?: hold=(?P<hold>\d))?"
)
MPC_RE = re.compile(
    r"\[MPC\] dt=(?P<dt>\d+)ms fishY=(?P<fishY>\d+) sCY=(?P<sCY>\d+) sH=(?P<sH>\d+) "
    r"sv=(?P<sv>-?\d+) fv=(?P<fv>-?\d+) cP=(?P<cP>\d+) cR=(?P<cR>\d+) hold=(?P<hold>\d)"
)
TRACK_RE = re.compile(r"\[ctrl\] track locked.*y=(?P<y>\d+).*h=(?P<h>\d+)")
STATE_RE = re.compile(r"\[vrchat_fish\] state (?P<from>\d) -> (?P<to>\d)")


@dataclass(frozen=True)
class CtrlFrame:
    file: str
    line_no: int
    detect_ms: int
    dt_ms: Optional[int]
    t_ms: Optional[int]
    mode: str  # fast/full
    fishY: int
    sCY: int
    sH: int
    src: str  # color/tpl
    hold: int  # last known holding from [MPC] lines (piecewise-constant)


@dataclass(frozen=True)
class Segment:
    file: str
    start_line: int
    end_line: int
    track_y: Optional[int]
    track_h: Optional[int]
    frames: List[CtrlFrame]
    mpc_dts: List[int]

    @property
    def track_top(self) -> Optional[int]:
        if self.track_y is None:
            return None
        return self.track_y + 30

    @property
    def track_bot(self) -> Optional[int]:
        if self.track_y is None or self.track_h is None:
            return None
        return self.track_y + self.track_h - 30


def _read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


def parse_segments(path: str) -> List[Segment]:
    lines = _read_text_lines(path)

    segments: List[Segment] = []
    current_hold = 0
    in_minigame = False
    seg_start = 0
    seg_track_y: Optional[int] = None
    seg_track_h: Optional[int] = None
    seg_frames: List[CtrlFrame] = []
    seg_mpc_dts: List[int] = []

    def flush_segment(end_line: int) -> None:
        nonlocal seg_start, seg_track_y, seg_track_h, seg_frames, seg_mpc_dts
        if seg_start <= 0:
            return
        segments.append(
            Segment(
                file=path,
                start_line=seg_start,
                end_line=end_line,
                track_y=seg_track_y,
                track_h=seg_track_h,
                frames=seg_frames,
                mpc_dts=seg_mpc_dts,
            )
        )
        seg_start = 0
        seg_track_y = None
        seg_track_h = None
        seg_frames = []
        seg_mpc_dts = []

    for idx, line in enumerate(lines, start=1):
        sm = STATE_RE.search(line)
        if sm:
            # state 2 -> 3 进入小游戏; state 3 -> 4 退出小游戏
            if sm.group("to") == "3":
                in_minigame = True
                current_hold = 0
                seg_start = idx
                seg_track_y = None
                seg_track_h = None
                seg_frames = []
                seg_mpc_dts = []
                continue
            if sm.group("from") == "3":
                in_minigame = False
                flush_segment(idx)
                continue

        if not in_minigame:
            continue

        tm = TRACK_RE.search(line)
        if tm:
            seg_track_y = int(tm.group("y"))
            seg_track_h = int(tm.group("h"))
            continue

        mm = MPC_RE.search(line)
        if mm:
            current_hold = int(mm.group("hold"))
            seg_mpc_dts.append(int(mm.group("dt")))
            continue

        cm = CTRL_RE.search(line)
        if cm:
            hold = cm.group("hold")
            dt_ms = cm.group("dt_ms")
            t_ms = cm.group("t_ms")
            seg_frames.append(
                CtrlFrame(
                    file=path,
                    line_no=idx,
                    detect_ms=int(cm.group("detect_ms")),
                    dt_ms=int(dt_ms) if dt_ms is not None else None,
                    t_ms=int(t_ms) if t_ms is not None else None,
                    mode=cm.group("mode"),
                    fishY=int(cm.group("fishY")),
                    sCY=int(cm.group("sCY")),
                    sH=int(cm.group("sH")),
                    src=cm.group("src"),
                    hold=int(hold) if hold is not None else current_hold,
                )
            )
            continue

    # 文件结尾仍在 minigame 的话也 flush
    flush_segment(len(lines))
    return segments


def _read_ini_section(path: str, section: str) -> dict:
    cur = None
    out: dict = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    cur = line[1:-1].strip()
                    continue
                if cur != section:
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except FileNotFoundError:
        return {}
    return out


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _solve_3x3(M: Sequence[Sequence[float]], b: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    # Gaussian elimination with partial pivoting.
    A = [
        [float(M[0][0]), float(M[0][1]), float(M[0][2]), float(b[0])],
        [float(M[1][0]), float(M[1][1]), float(M[1][2]), float(b[1])],
        [float(M[2][0]), float(M[2][1]), float(M[2][2]), float(b[2])],
    ]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            return None
        A[col], A[pivot] = A[pivot], A[col]

        piv = A[col][col]
        for j in range(col, 4):
            A[col][j] /= piv

        for r in range(3):
            if r == col:
                continue
            factor = A[r][col]
            if factor == 0.0:
                continue
            for j in range(col, 4):
                A[r][j] -= factor * A[col][j]
    return (A[0][3], A[1][3], A[2][3])


def _fit_linear_xy(samples: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    # Fit y = a*x + b, return (a, b, rmse).
    n = len(samples)
    if n < 3:
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    if vx < 1e-12:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    a = cov / vx
    b = my - a * mx
    rmse = math.sqrt(sum((ys[i] - (a * xs[i] + b)) ** 2 for i in range(n)) / n)
    return (a, b, rmse)


def _fit_joint(samples: Sequence[Tuple[float, float, int]]) -> Optional[Tuple[float, float, float]]:
    # y = drag*x + gravity*(1-hold) + thrust*(hold)
    # params = [drag, gravity, thrust]
    M = [[0.0, 0.0, 0.0] for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    for x, y, hold in samples:
        a0 = float(x)
        a1 = float(1 - hold)
        a2 = float(hold)
        a = (a0, a1, a2)
        b[0] += a0 * y
        b[1] += a1 * y
        b[2] += a2 * y
        for i in range(3):
            for j in range(3):
                M[i][j] += a[i] * a[j]
    return _solve_3x3(M, b)


def _rmse(samples: Sequence[Tuple[float, float, int]], params: Tuple[float, float, float]) -> float:
    drag, gravity, thrust = params
    err2 = 0.0
    for x, y, hold in samples:
        accel = thrust if hold else gravity
        pred = drag * x + accel
        e = y - pred
        err2 += e * e
    return math.sqrt(err2 / max(1, len(samples)))


def _median_abs_err(samples: Sequence[Tuple[float, float, int]], params: Tuple[float, float, float]) -> float:
    drag, gravity, thrust = params
    abs_err = []
    for x, y, hold in samples:
        accel = thrust if hold else gravity
        pred = drag * x + accel
        abs_err.append(abs(y - pred))
    return statistics.median(abs_err) if abs_err else float("nan")


def _trimmed_refit(
    samples: Sequence[Tuple[float, float, int]],
    trim_ratio: float,
) -> Tuple[Optional[Tuple[float, float, float]], Sequence[Tuple[float, float, int]]]:
    if not samples:
        return (None, samples)
    if trim_ratio <= 0.0:
        return (_fit_joint(samples), samples)

    base = _fit_joint(samples)
    if base is None:
        return (None, samples)

    drag, gravity, thrust = base
    scored: List[Tuple[float, Tuple[float, float, int]]] = []
    for s in samples:
        x, y, hold = s
        accel = thrust if hold else gravity
        pred = drag * x + accel
        scored.append((abs(y - pred), s))
    scored.sort(key=lambda t: t[0])
    keep_n = max(10, int(round(len(scored) * (1.0 - trim_ratio))))
    kept = [s for _, s in scored[:keep_n]]
    return (_fit_joint(kept), kept)


def _clamp_dt_ms(dt_ms: float) -> float:
    if dt_ms < 1.0:
        return 1.0
    if dt_ms > 1000.0:
        return 1000.0
    return dt_ms


def _interval_dt_ms(prev: CtrlFrame, cur: CtrlFrame, assume_dt_ms: float) -> float:
    """
    计算 prev->cur 的时间间隔（ms），优先使用 ctrl 行里的 t= 时间戳。
    若无时间戳，则退化使用 cur.dt_ms（注意：过滤掉中间帧时此值可能偏差）。
    """
    if prev.t_ms is not None and cur.t_ms is not None:
        dt = cur.t_ms - prev.t_ms
        if dt > 0:
            return _clamp_dt_ms(float(dt))
    if cur.dt_ms is not None and cur.dt_ms > 0:
        return _clamp_dt_ms(float(cur.dt_ms))
    return _clamp_dt_ms(float(assume_dt_ms))


def build_triples(
    seg: Segment,
    *,
    base_dt_ms: float,
    assume_dt_ms: float,
    min_slider_h: int,
    include_tpl: bool,
    require_hold_stable: int,
    max_jump_px: int,
    max_line_gap: int,
    exclude_boundary_px: int,
) -> List[Tuple[float, float, int]]:
    track_top = seg.track_top
    track_bot = seg.track_bot

    # 先按 src/sliderH 过滤
    frames = [
        f
        for f in seg.frames
        if (include_tpl or f.src == "color") and f.sH >= min_slider_h
    ]

    triples: List[Tuple[float, float, int]] = []
    for i in range(len(frames) - 2):
        f0, f1, f2 = frames[i], frames[i + 1], frames[i + 2]

        # CtrlFrame.hold 是“上一段时间内实际施加的动作”。
        # 三帧组 (f0,f1,f2) 中:
        #   d1 = f1 - f0 ≈ v1  (由 f1.hold 对应的动作产生)
        #   d2 = f2 - f1 ≈ v2  (由 f2.hold 对应的动作产生)
        # 拟合方程: d2 = d1 * drag + accel(动作=f2.hold)
        if require_hold_stable == 2 and f1.hold != f2.hold:
            continue
        if require_hold_stable == 3 and not (f0.hold == f1.hold == f2.hold):
            continue
        hold = f2.hold

        # 粗略跳变过滤：按“像素跳变”过滤，不受 dt 缩放影响
        d1_px = f1.sCY - f0.sCY
        d2_px = f2.sCY - f1.sCY
        if abs(d1_px) > max_jump_px or abs(d2_px) > max_jump_px:
            continue

        # 跳过明显的长间隔（通常是 MISS/状态切换等）
        if (f1.line_no - f0.line_no) > max_line_gap or (f2.line_no - f1.line_no) > max_line_gap:
            continue

        # 过滤边界附近样本（防止反弹影响 drag/accel 拟合）
        if track_top is not None and track_bot is not None and exclude_boundary_px > 0:
            if (
                f0.sCY <= track_top + exclude_boundary_px
                or f0.sCY >= track_bot - exclude_boundary_px
                or f1.sCY <= track_top + exclude_boundary_px
                or f1.sCY >= track_bot - exclude_boundary_px
                or f2.sCY <= track_top + exclude_boundary_px
                or f2.sCY >= track_bot - exclude_boundary_px
            ):
                continue

        dt01 = _interval_dt_ms(f0, f1, assume_dt_ms)
        dt12 = _interval_dt_ms(f1, f2, assume_dt_ms)
        d1 = d1_px * base_dt_ms / dt01
        d2 = d2_px * base_dt_ms / dt12
        triples.append((d1, d2, hold))
    return triples


def estimate_bounce_e(
    seg: Segment,
    *,
    base_dt_ms: float,
    assume_dt_ms: float,
    min_slider_h: int,
    include_tpl: bool,
    params: Tuple[float, float, float],
    boundary_eps_px: int,
    max_jump_px: int,
    max_line_gap: int,
) -> List[float]:
    """
    用边界碰撞事件粗略估计反弹系数 e。

    估计假设（近似）:
      v_out = (-e * v_in) * drag + accel_next
    其中 v_in = v_prev * drag + accel_prev

    由于日志缺少“每步 hold/dt”，这里要求碰撞事件前后 hold 分段不变，
    且下一帧不在边界上，尽量减少夹紧/二次碰撞干扰。
    """
    track_top = seg.track_top
    track_bot = seg.track_bot
    if track_top is None or track_bot is None:
        return []

    drag, gravity, thrust = params

    frames = [
        f
        for f in seg.frames
        if (include_tpl or f.src == "color") and f.sH >= min_slider_h
    ]

    def accel(hold: int) -> float:
        return thrust if hold else gravity

    out: List[float] = []
    for i in range(2, len(frames) - 2):
        f_prev2, f_prev, f, f_next = frames[i - 2], frames[i - 1], frames[i], frames[i + 1]

        # 跳变 / 长间隔过滤
        if (
            abs(f_prev.sCY - f_prev2.sCY) > max_jump_px
            or abs(f.sCY - f_prev.sCY) > max_jump_px
            or abs(f_next.sCY - f.sCY) > max_jump_px
        ):
            continue
        if (
            (f_prev.line_no - f_prev2.line_no) > max_line_gap
            or (f.line_no - f_prev.line_no) > max_line_gap
            or (f_next.line_no - f.line_no) > max_line_gap
        ):
            continue

        at_top = abs(f.sCY - track_top) <= boundary_eps_px
        at_bot = abs(f.sCY - track_bot) <= boundary_eps_px
        if not (at_top or at_bot):
            continue

        # 下一帧仍在边界 -> 很可能在“贴边抖动/连续碰撞”，不适合估计
        if abs(f_next.sCY - track_top) <= boundary_eps_px or abs(f_next.sCY - track_bot) <= boundary_eps_px:
            continue

        # 要求 hold 在这一小段稳定，减少“未知 action 变化”带来的误差
        if not (f_prev2.hold == f_prev.hold == f.hold == f_next.hold):
            continue

        dt_prev = _interval_dt_ms(f_prev2, f_prev, assume_dt_ms)
        v_prev = (f_prev.sCY - f_prev2.sCY) * base_dt_ms / dt_prev
        v_in = v_prev * drag + accel(f.hold)

        # v_in 必须确实是“撞向边界”
        if at_top and v_in >= -1e-6:
            continue
        if at_bot and v_in <= 1e-6:
            continue

        dt_out = _interval_dt_ms(f, f_next, assume_dt_ms)
        v_out_obs = (f_next.sCY - f.sCY) * base_dt_ms / dt_out
        a_next = accel(f_next.hold)

        denom = drag * v_in
        if abs(denom) < 1e-6:
            continue

        e = -(v_out_obs - a_next) / denom
        if 0.0 <= e <= 1.5:
            out.append(e)
    return out


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs",
        nargs="*",
        help="log.csv文件路径（默认自动找 data/logs/log.csv 或 log.csv）",
    )
    parser.add_argument("--config", default="config.ini", help="用于对比当前配置的 config.ini 路径")

    parser.add_argument("--base-dt-ms", type=float, default=50.0, help="代码里的 BASE_DT_MS（默认 50）")
    parser.add_argument(
        "--assume-dt-ms",
        type=float,
        default=0.0,
        help="假设每步实际 dt（0 表示使用日志中 [MPC] 的 dt 中位数）",
    )

    parser.add_argument("--min-slider-h", type=int, default=80, help="过滤 sH 过小的帧")
    parser.add_argument(
        "--include-tpl",
        action="store_true",
        help="也使用 [tpl] 帧（默认只用 [color]）",
    )
    parser.add_argument(
        "--require-hold-stable",
        type=int,
        choices=[2, 3],
        default=3,
        help="三帧组内 hold 稳定性要求：2=要求 f1.hold==f2.hold；3=要求 f0==f1==f2（默认）",
    )

    parser.add_argument("--max-jump-px", type=int, default=100, help="过滤单步像素跳变过大的样本")
    parser.add_argument("--max-line-gap", type=int, default=8, help="过滤 log 行号间隔过大的样本")
    parser.add_argument("--exclude-boundary-px", type=int, default=2, help="过滤靠近边界的样本（避免反弹影响）")
    parser.add_argument(
        "--trim",
        type=float,
        default=0.05,
        help="截尾比例（例如 0.05 表示去掉最大 5%% 残差样本后再拟合一次）",
    )

    parser.add_argument(
        "--fit-bounce",
        action="store_true",
        help="额外用碰撞事件粗略估计反弹系数 e（需要 track locked 且边界事件足够）",
    )
    parser.add_argument("--bounce-eps-px", type=int, default=1, help="判定“在边界”的像素容差")
    parser.add_argument("--verbose", action="store_true", help="打印每段统计信息")

    args = parser.parse_args(list(argv))

    logs = list(args.logs)
    if not logs:
        candidates = [
            "data/logs/log.csv",
            "log.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                logs = [c]
                break
        if not logs:
            print("未指定日志文件，且未找到默认的 log.csv", file=sys.stderr)
            return 2

    all_segments: List[Segment] = []
    for p in logs:
        segs = parse_segments(p)
        if not segs:
            print(f"[warn] 未在 {p} 中找到 state 2->3 / 3->4 的 minigame 段")
        all_segments.extend(segs)

    if not all_segments:
        print("没有可用的 minigame 段，无法拟合", file=sys.stderr)
        return 2

    # dt 统计
    all_dts: List[int] = []
    for s in all_segments:
        all_dts.extend(s.mpc_dts)
    dt_p50 = statistics.median(all_dts) if all_dts else args.base_dt_ms
    assume_dt_ms = args.assume_dt_ms if args.assume_dt_ms > 0.0 else float(dt_p50)
    if assume_dt_ms <= 0.0:
        assume_dt_ms = args.base_dt_ms

    # 读取当前 config 做对比（可选）
    vr_cfg = _read_ini_section(args.config, "vrchat_fish")
    cur_drag = _as_float(vr_cfg.get("bb_drag"))
    cur_grav = _as_float(vr_cfg.get("bb_gravity"))
    cur_thr = _as_float(vr_cfg.get("bb_thrust"))

    # 构建 triples
    triples: List[Tuple[float, float, int]] = []
    for seg in all_segments:
        seg_triples = build_triples(
            seg,
            base_dt_ms=args.base_dt_ms,
            assume_dt_ms=assume_dt_ms,
            min_slider_h=args.min_slider_h,
            include_tpl=args.include_tpl,
            require_hold_stable=args.require_hold_stable,
            max_jump_px=args.max_jump_px,
            max_line_gap=args.max_line_gap,
            exclude_boundary_px=args.exclude_boundary_px,
        )
        triples.extend(seg_triples)
        if args.verbose:
            print(
                f"[seg] {seg.file}:{seg.start_line}-{seg.end_line} "
                f"frames={len(seg.frames)} triples={len(seg_triples)} "
                f"track_top={seg.track_top} track_bot={seg.track_bot} mpc_dt_n={len(seg.mpc_dts)}"
            )

    if not triples:
        print("过滤后没有 triples 样本，无法拟合（可以尝试降低过滤阈值）", file=sys.stderr)
        return 2

    hold_xy = [(x, y) for (x, y, h) in triples if h == 1]
    rel_xy = [(x, y) for (x, y, h) in triples if h == 0]

    sep_hold = _fit_linear_xy(hold_xy)
    sep_rel = _fit_linear_xy(rel_xy)

    joint = _fit_joint(triples)
    if joint is None:
        print("联合拟合失败（矩阵奇异），请检查数据过滤条件", file=sys.stderr)
        return 2

    joint_rmse = _rmse(triples, joint)
    joint_med = _median_abs_err(triples, joint)

    trimmed_params, kept = _trimmed_refit(triples, args.trim)
    if trimmed_params is None:
        trimmed_params = joint
        kept = triples

    trimmed_rmse = _rmse(kept, trimmed_params)
    trimmed_med = _median_abs_err(kept, trimmed_params)

    print("=== 输入统计 ===")
    print(f"logs: {', '.join(logs)}")
    print(f"segments: {len(all_segments)}")
    print(f"[MPC] dt samples: {len(all_dts)}  p50={dt_p50:.0f}ms  min={min(all_dts) if all_dts else 'n/a'}  max={max(all_dts) if all_dts else 'n/a'}")
    print(f"base_dt_ms={args.base_dt_ms:.0f}  assume_dt_ms={assume_dt_ms:.0f}  scale={args.base_dt_ms/assume_dt_ms:.3f}")
    print(f"triples: {len(triples)}  hold={len(hold_xy)}  release={len(rel_xy)}")
    print()

    print("=== 分开拟合（诊断用，drag 可能不同） ===")
    if sep_hold:
        d, thr, rmse = sep_hold
        print(f"hold=1: drag={d:.4f} thrust={thr:.4f} rmse={rmse:.2f}")
    else:
        print("hold=1: not enough data")
    if sep_rel:
        d, grav, rmse = sep_rel
        print(f"hold=0: drag={d:.4f} gravity={grav:.4f} rmse={rmse:.2f}")
    else:
        print("hold=0: not enough data")
    print()

    print("=== 联合拟合（共享 drag） ===")
    d, g, t = joint
    print(f"drag={d:.4f} gravity={g:.4f} thrust={t:.4f}")
    print(f"rmse={joint_rmse:.2f}  median|err|={joint_med:.2f}")
    print()

    print(f"=== 截尾再拟合（trim={args.trim:.2f}） ===")
    d, g, t = trimmed_params
    print(f"drag={d:.4f} gravity={g:.4f} thrust={t:.4f}")
    print(f"kept={len(kept)}/{len(triples)}  rmse={trimmed_rmse:.2f}  median|err|={trimmed_med:.2f}")
    print()

    if cur_drag is not None or cur_grav is not None or cur_thr is not None:
        print("=== 当前 config.ini（对比） ===")
        if cur_drag is not None:
            print(f"bb_drag={cur_drag}")
        if cur_grav is not None:
            print(f"bb_gravity={cur_grav}")
        if cur_thr is not None:
            print(f"bb_thrust={cur_thr}")
        # 用当前 config 参数评估单步误差
        if cur_drag is not None and cur_grav is not None and cur_thr is not None:
            cur_params = (cur_drag, cur_grav, cur_thr)
            cur_rmse = _rmse(triples, cur_params)
            cur_med = _median_abs_err(triples, cur_params)
            print(f"rmse={cur_rmse:.2f}  median|err|={cur_med:.2f}")
        print()

    # ── 多步轨迹网格搜索（优化 MPC 实际使用的多步预测误差） ──
    print("=== 多步轨迹网格搜索 ===")
    print("（MPC 使用 sim_horizon 步预测，单步最优参数不一定多步最优）")

    # 构建连续帧序列（每个 segment 独立）
    HORIZON = 1  # 与 config 的 bb_sim_horizon 一致

    def _multistep_mean_err(
        segments: Sequence[Segment],
        params: Tuple[float, float, float],
        horizon: int,
        base_dt: float,
        assume_dt: float,
        min_sH: int,
        include_tpl_: bool,
        max_jump: int,
        max_gap: int,
    ) -> Tuple[float, int]:
        """对所有 segment 评估多步预测误差，返回 (mean_abs_err, n_eval)。"""
        drag, gravity, thrust = params
        total_err = 0.0
        n_eval = 0
        for seg in segments:
            frames = [
                f for f in seg.frames
                if (include_tpl_ or f.src == "color") and f.sH >= min_sH
            ]
            if len(frames) < horizon + 2:
                continue
            for i in range(1, len(frames) - horizon):
                f_prev = frames[i - 1]
                f_start = frames[i]
                # 连续性检查
                ok = True
                for j in range(horizon):
                    f_next = frames[i + 1 + j]
                    if f_next.line_no - frames[i + j].line_no > max_gap:
                        ok = False
                        break
                    if abs(f_next.sCY - frames[i + j].sCY) > max_jump:
                        ok = False
                        break
                if not ok:
                    continue
                # 初始速度
                dt0 = _interval_dt_ms(f_prev, f_start, assume_dt)
                v0 = (f_start.sCY - f_prev.sCY) * base_dt / dt0
                # 模拟（MPC 风格：不做 dt 缩放，每步 = base_dt）
                y = float(f_start.sCY)
                v = v0
                for j in range(horizon):
                    f_next = frames[i + 1 + j]
                    accel = thrust if f_next.hold else gravity
                    v = v * drag + accel
                    y = y + v
                actual_y = frames[i + horizon].sCY
                total_err += abs(y - actual_y)
                n_eval += 1
        if n_eval == 0:
            return (float("inf"), 0)
        return (total_err / n_eval, n_eval)

    # 先评估当前 config 和截尾拟合在多步上的表现
    candidates = []
    if cur_drag is not None and cur_grav is not None and cur_thr is not None:
        cur_params = (cur_drag, cur_grav, cur_thr)
        err_cur, n_cur = _multistep_mean_err(
            all_segments, cur_params, HORIZON, args.base_dt_ms, assume_dt_ms,
            args.min_slider_h, args.include_tpl, args.max_jump_px, args.max_line_gap)
        print(f"  config.ini:  drag={cur_drag:.2f} grav={cur_grav:.2f} thr={cur_thr:.2f}  "
              f"h{HORIZON} mean|err|={err_cur:.1f}  n={n_cur}")
        candidates.append(("config", cur_params, err_cur))

    err_trim, n_trim = _multistep_mean_err(
        all_segments, trimmed_params, HORIZON, args.base_dt_ms, assume_dt_ms,
        args.min_slider_h, args.include_tpl, args.max_jump_px, args.max_line_gap)
    print(f"  trimmed:     drag={trimmed_params[0]:.4f} grav={trimmed_params[1]:.4f} thr={trimmed_params[2]:.4f}  "
          f"h{HORIZON} mean|err|={err_trim:.1f}  n={n_trim}")
    candidates.append(("trimmed", trimmed_params, err_trim))

    # 网格搜索
    print(f"\n  grid search (horizon={HORIZON})...")
    best_err = float("inf")
    best_params: Optional[Tuple[float, float, float]] = None
    grid_n = 0
    for drag_i in range(50, 100, 2):        # 0.50 ~ 0.98
        drag_v = drag_i / 100.0
        for grav_i in range(25, 250, 25):     # 0.25 ~ 2.25
            grav_v = grav_i / 100.0
            for thr_i in range(-400, -50, 25): # -4.00 ~ -0.75
                thr_v = thr_i / 100.0
                grid_n += 1
                err, n = _multistep_mean_err(
                    all_segments, (drag_v, grav_v, thr_v), HORIZON,
                    args.base_dt_ms, assume_dt_ms,
                    args.min_slider_h, args.include_tpl, args.max_jump_px, args.max_line_gap)
                if err < best_err:
                    best_err = err
                    best_params = (drag_v, grav_v, thr_v)

    if best_params is not None:
        print(f"  grid best:   drag={best_params[0]:.2f} grav={best_params[1]:.2f} thr={best_params[2]:.2f}  "
              f"h{HORIZON} mean|err|={best_err:.1f}  (searched {grid_n} combos)")
        candidates.append(("grid", best_params, best_err))

        # 在 grid best 附近做精细搜索
        d0, g0, t0 = best_params
        fine_best_err = best_err
        fine_best = best_params
        for dd in range(-5, 6):           # ±0.05
            drag_v = d0 + dd * 0.01
            if drag_v < 0.3 or drag_v > 1.0:
                continue
            for dg in range(-10, 11):      # ±0.50
                grav_v = g0 + dg * 0.05
                if grav_v < 0.0 or grav_v > 5.0:
                    continue
                for dt_ in range(-10, 11): # ±0.50
                    thr_v = t0 + dt_ * 0.05
                    if thr_v > 0.0 or thr_v < -6.0:
                        continue
                    err, n = _multistep_mean_err(
                        all_segments, (drag_v, grav_v, thr_v), HORIZON,
                        args.base_dt_ms, assume_dt_ms,
                        args.min_slider_h, args.include_tpl, args.max_jump_px, args.max_line_gap)
                    if err < fine_best_err:
                        fine_best_err = err
                        fine_best = (drag_v, grav_v, thr_v)

        if fine_best != best_params:
            print(f"  fine-tuned:  drag={fine_best[0]:.2f} grav={fine_best[1]:.2f} thr={fine_best[2]:.2f}  "
                  f"h{HORIZON} mean|err|={fine_best_err:.1f}")
            candidates.append(("fine", fine_best, fine_best_err))

    # 选出最优
    candidates.sort(key=lambda c: c[2])
    winner_name, winner_params, winner_err = candidates[0]
    print()

    print("=== 推荐写入 config.ini（vrchat_fish 段） ===")
    print(f"# 来源: {winner_name}  h{HORIZON} mean|err|={winner_err:.1f}")
    print(f"bb_drag={winner_params[0]:.2f}")
    print(f"bb_gravity={winner_params[1]:.2f}")
    print(f"bb_thrust={winner_params[2]:.2f}")
    print()

    if args.fit_bounce:
        es: List[float] = []
        for seg in all_segments:
            es.extend(
                estimate_bounce_e(
                    seg,
                    base_dt_ms=args.base_dt_ms,
                    assume_dt_ms=assume_dt_ms,
                    min_slider_h=args.min_slider_h,
                    include_tpl=args.include_tpl,
                    params=trimmed_params,
                    boundary_eps_px=args.bounce_eps_px,
                    max_jump_px=args.max_jump_px,
                    max_line_gap=args.max_line_gap,
                )
            )
        if es:
            p50 = statistics.median(es)
            p10 = statistics.quantiles(es, n=10)[0] if len(es) >= 10 else min(es)
            p90 = statistics.quantiles(es, n=10)[-1] if len(es) >= 10 else max(es)
            print("=== 反弹系数估计（粗略） ===")
            print(f"bounce_e: n={len(es)}  p50={p50:.3f}  p10={p10:.3f}  p90={p90:.3f}")
            print("（代码里目前是写死的 0.8；如果你准备做成可配参数，这个值可以做参考。）")
        else:
            print("=== 反弹系数估计（粗略） ===")
            print("未找到足够的边界碰撞事件用于估计（或被过滤条件排掉了）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
