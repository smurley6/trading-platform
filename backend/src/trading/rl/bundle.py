"""Assemble the static website demo bundle ``demo.json`` (task-rl-12; program doc 2026-10-21
retarget addendum, milestone W0).

PURE half of ``trading.rl``: stdlib + first-party pure modules only (``domain.rl_features``,
``domain.rl_policy``, ``application.rl_evaluation``, ``rl.costs``) — no numpy, no gymnasium, no
torch — so the bundle's own arithmetic (the stdlib replay, the equity accounting, the stitching,
the bands, the float formatting) runs with zero optional deps. ``jobs/rl_bundle.py`` produces the
env replays (``trading.rl.scoring``) this module stitches, and cross-checks this module's stdlib
replay against them; nothing here steps an environment.

**What the bundle is** (the schema is ``docs/demo-bundle.schema.json``, JSON Schema 2020-12, which
the website repo vendors): ``meta`` (provenance + the price-label guardrail), ``bars`` (every
``{date, close}`` of the frozen dataset), ``features`` (the ``domain/rl_features`` spec the
TypeScript mirror rebuilds observations with), ``policy`` (the EXHIBIT seed's per-fold policies +
``parity_vectors``), ``replay`` (the exhibit seed's per-day decisions over the walk-forward TEST
segments), ``curves`` (stitched test-segment equity per cost, train/val/purge only as flagged
background spans) and ``eval_table`` (look #N's statistics, transformed — never re-run).

**The exhibit is the EVALUATED agent, not one network** (task-rl-10 review addendum, option (a)).
The eval table's agent is 5 seeds x K folds of per-fold policies, each trained on exactly its
fold's ``train`` window. The exhibit replays ONE designated seed (``meta.exhibit_seed``) by
stitching that seed's per-fold policies across their own test segments: fold *k*'s policy decides
every bar of fold *k*'s test window and nothing else. Every headline statistic stays the >= 5-seed,
fold-clustered distribution in ``eval_table`` (exhibit-seed honesty rule).

**Execution semantics** (exactly the env's lag-0 timing contract, ``rl/env.py``): at decision bar
``t`` the policy sees the observation built from ``closes[: t + 1]`` and the held position; a legal
BUY (when flat) / SELL (when long) fills at close ``t + 1``; an illegal one is a HOLD no-op. Each
step books ``E_{t+1} = E_t * ((1 + p_t * (C_{t+1} / C_t - 1)) * (1 - (bps * 1e-4) * |p_{t+1} -
p_t|))`` with ``p_t`` the position held over ``t -> t+1`` and ``p_{t+1}`` the position after the
fill at ``C_{t+1}`` (:func:`equity_path`). Each fold's episode starts FLAT at equity 1.0 (as rl-10
evaluated it); the stitched curve chain-links the per-fold episodes, so a fold boundary charges no
exit cost for a position still open at the fold's last mark and the next fold re-enters (paying its
entry cost) — the per-fold accounting, unchanged.

**Float formatting (deterministic; byte-identical reruns are an acceptance test).** The writer is
``json.dumps(sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)`` and every
float is emitted with Python's shortest round-trip ``repr`` of a value pre-rounded here:

- ``f32-b64`` (policy weights/biases): the array's float32 values, little-endian, BYTE-PLANE
  SHUFFLED (all ``n`` lowest bytes, then all second bytes, third, fourth — the layout that lets
  gzip squeeze the sign/exponent planes), standard padded base64 (RFC 4648). LOSSLESS: the
  exported weights ARE float32 values (SB3 trains in float32; the export widens them exactly).
  Chosen by the size budget: all 21 exhibit nets as decimal text are ~510 KB gzip on their own,
  this encoding ~383 KB (:func:`encode_f32` / :func:`decode_f32`).
- ``f32`` (parity-vector observations): the shortest decimal (1..9 significant digits) that parses
  back to the SAME float32 — lossless; recover it with ``Math.fround(parsed)``.
- ``sig11`` (equity curves): rounded to 11 significant digits (relative error <= 5e-11).
- ``sig6`` (display only: replay ``action_probs``, the random bands, the seed envelope): 6
  significant digits. ``curves.formats`` names each series' format.
- ``exact`` (closes, parity logits/probs, eval-table statistics, costs): the float64 ``repr``,
  unrounded.

**Curves honesty (PIT review 2026-10-22).** The exhibit seed is the BEST of the 5 on this span
(``meta.exhibit_rank``: every seed's stitched final equity and mean per-fold sharpe per cost, from
the report's rows), so the label says so, and the curves shade the full 5-seed envelope
(``agent_seeds_min`` / ``agent_seeds_max``) around it. The HEADLINE band is the TURNOVER-MATCHED
random policy (``random_matched_*``: rl-10's ``random_matched[ppo]``, the null GATE-LEARN failed to
beat), not the uniform random policy, which trades more and pays more cost — an agent line above a
uniform band reads as "beats random" while the gate says otherwise. The uniform band
(``random_uniform_*``) ships only while the size budget allows; it is dropped first.

**Returns-only (program doc §4.2 BINDING rule).** ``bars[].close`` are ``adjustment=all`` closes:
ratios are point-in-time total returns, LEVELS are back-adjusted with later dividends. Every number
the bundle derives from them is a ratio (equity, features); the page must label any displayed
level with ``meta.price_label`` — never "SPY's price".
"""

import base64
import binascii
import gzip
import json
import math
import re
import statistics
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from trading.application.rl_evaluation import (
    VS_BUY_AND_HOLD,
    GateVerdict,
    WalkForwardFold,
    buy_and_hold_line,
    two_stage_bootstrap_iqm_ci,
)
from trading.domain.rl_features import (
    FEATURE_LOOKBACKS,
    MAX_LOOKBACK,
    OBS_HIGH,
    OBS_LOW,
    build_observation,
)
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy, RlAction
from trading.rl.costs import BPS

BUNDLE_SCHEMA_VERSION: Final = 1
BUNDLE_KIND: Final = "rl.demo.bundle"
SCHEMA_FILE: Final = "docs/demo-bundle.schema.json"
MAX_RAW_BYTES: Final = 2_500_000
MAX_GZIP_BYTES: Final = 600_000
MIN_PARITY_VECTORS: Final = 20
CURVE_DIGITS: Final = 11
BAND_DIGITS: Final = 6
PROB_DIGITS: Final = 6
# Series names in ``curves.by_cost[cost]`` (report policy ids where one exists).
AGENT_SERIES: Final = "agent"
ENVELOPE_SERIES: Final = ("agent_seeds_min", "agent_seeds_max")
BASELINE_SERIES: Final = ("buy_and_hold", "always_flat", "sma_cross_20_50")
# The HEADLINE band: the turnover-matched random policy, the null GATE-LEARN is judged against.
MATCHED_BAND_SERIES: Final = ("random_matched_p25", "random_matched_p50", "random_matched_p75")
# Optional (dropped first when the size budget forces a choice): the uniform 1/3 random policy.
UNIFORM_BAND_SERIES: Final = ("random_uniform_p25", "random_uniform_p50", "random_uniform_p75")
MATCHED_BAND_LABEL: Final = "turnover-matched random (the GATE-LEARN null)"
UNIFORM_BAND_LABEL: Final = "uniform random (trades more; not the GATE-LEARN null)"
# Baselines with ONE run per fold whose IQM CI rl-12 may recompute from the report's own rows
# (task-rl-10 review addendum: fold-only bootstrap, the same estimator rl-10 now reports).
CI_RECOMPUTABLE: Final = frozenset(BASELINE_SERIES)
PRICE_LABEL_TEMPLATE: Final = (
    "dividend-adjusted (total-return) closes as served by Alpaca on {generated_at}; may reflect "
    "later corrections"
)
EXHIBIT_LABEL_TEMPLATE: Final = (
    "exhibit seed {seed}: ONE training seed's per-fold policies, each deciding only its own "
    "fold's out-of-sample test window (stitched). {standing} It is an illustration, not the "
    "result: every statistic in eval_table is the {n_seeds}-seed, fold-clustered distribution "
    "over {n_folds} folds"
)
# The report's IQM-CI note predates the fold-only CI for single-run baselines; once rl-12 has
# recomputed those CIs the note would contradict the rows beside it, so it is replaced.
STALE_CI_NOTE_NEEDLE: Final = "one run per fold has no CI"
CI_NOTE_REPLACEMENT: Final = (
    "IQM CIs are two-stage bootstraps (folds resampled, then seeds within each drawn fold); for a "
    "policy with one run per fold the inner stage is a no-op, so it carries a fold-only CI of the "
    "same estimator (recomputed by rl-12 from the report's per-fold rows; see bundle_notes). The "
    "gates read the fold-clustered CR0 SE with t(K-1) [rl-12: replaces the report's note, which "
    "predates those CIs]"
)
RECOMPUTED_CI_NOTE: Final = "recomputed from per-fold rows (fold-only bootstrap)"
CI_RECOMPUTE_NOTE: Final = (
    "sharpe IQM CIs marked sharpe_iqm_ci_source='recomputed' were absent (None) in the evaluation "
    "report (its code then gave single-run baselines no CI); rl-12 recomputed them from the "
    "report's OWN per-fold sharpe rows with application.rl_evaluation.two_stage_bootstrap_iqm_ci "
    "(fold-only bootstrap, 2000 draws, seed 0 — the estimator rl-10 now reports for them). No "
    "test-window number was replayed for this; it is not a new look. Every CI the report did "
    "record is copied verbatim (and re-derived identically as a check)"
)
_FORMATS: Final = {
    "f32-b64": "an array of n float32 values: little-endian bytes, byte-plane shuffled (byte 0 "
    "of every value, then byte 1, byte 2, byte 3; value i's byte j is at offset j * n + i), then "
    "standard padded base64 (RFC 4648). LOSSLESS. Matrices are row-major [out][in]",
    "f32": "shortest decimal (1..9 significant digits) that parses back to the same float32; "
    "LOSSLESS for float32 values — recover the exact value with Math.fround(parsed)",
    "sig11": "rounded to 11 significant digits (relative error <= 5e-11)",
    "sig6": "rounded to 6 significant digits (display only: replay probabilities, curve bands "
    "and the seed envelope)",
    "exact": "float64 shortest round-trip repr, unrounded",
    "writer": "json.dumps(sort_keys=True, separators=(',', ':'), allow_nan=False, "
    "ensure_ascii=True) + '\\n' (UTF-8); deterministic: equal inputs => equal bytes",
}
_FEATURE_FORMULAS: Final = (
    "ln(C_t / C_{t-1}) / s.daily_vol",
    "ln(C_t / C_{t-5}) / (s.daily_vol * sqrt(5))",
    "ln(C_t / C_{t-21}) / (s.daily_vol * sqrt(21))",
    "(C_t / SMA_10 - 1) / s.sma10_gap",
    "(C_t / SMA_50 - 1) / s.sma50_gap",
    "vol_21 / s.daily_vol - 1",
    "(SMA_10 / SMA_50 - 1) / s.trend_gap",
    "1.0 if a position is held else 0.0",
    "(C_t / entry_close - 1) / s.unrealized; 0.0 when flat",
)
_ARITHMETIC: Final = (
    "SMA_n = left-to-right float sum of the last n closes C_{t-n+1}..C_t, divided by n",
    "vol_21 = SAMPLE standard deviation (divide by 20) of the 21 log returns ln(C_i / C_{i-1}), "
    "i = t-20..t: mean = left-to-right sum / 21; variance = left-to-right sum of squared "
    "deviations / 20; sqrt",
    "every value is clipped to [obs_low, obs_high], then rounded to float32 (Math.fround) "
    "before the forward pass — the exact inputs the policy was trained and parity-checked on",
    "entry_close = the close at which the open position was FILLED (bar t_buy + 1), never the "
    "decision bar's close",
    "ln/tanh are not bit-identical across libm/V8 (~1 ulp): compare with a tolerance",
)
_POLICY_FORWARD: Final = (
    "each unit = (left-to-right sum over j of weight[i][j] * x[j]) + bias[i], then the "
    "activation (tanh / identity); logits = action_head(hidden...(obs)); action = "
    "actions[argmax(logits)], an exact tie resolving to the LOWEST index; never sampled. "
    "action_probs = softmax: m = max(logits); e_i = exp(l_i - m); p_i = e_i / (left-to-right "
    "sum of e)"
)

# Key-like / account-like / local-path-like strings (acceptance test 5). Alpaca key NAMES are
# matched with their separator (``APCA-API-KEY-ID`` / ``APCA_API_SECRET_KEY``): a bare "APCA" can
# occur by chance inside the base64 weight blobs, a "-" or "_" cannot (not in the base64 alphabet).
_SECRET_PATTERNS: Final = (
    re.compile(r"APCA[-_]"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}"),
    re.compile(r"Bearer\s+\S"),
    re.compile(r"postgres(?:ql)?(?:\+\w+)?://(?!(?:[^/@\s]*@)?(?:localhost|127\.0\.0\.1)\b)"),
    re.compile(r"\b(?:PK|AK|CK)[A-Z0-9]{16,}\b"),
    re.compile(r"(?<![\w.])/(?:Users|home|private|var/folders|tmp)/"),
    re.compile(r"\b[A-Za-z]:\\"),
    re.compile(r"(?<![\w.])/(?:opt|Volumes|root)/"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"API_KEY\s*[=:]"),
)


class BundleError(ValueError):
    """The bundle cannot be assembled honestly from these inputs (fails closed, never guesses)."""


# --- float formats --------------------------------------------------------------------------------


def fround(value: float) -> float:
    """Round to the nearest float32 (IEEE round-half-even), widened back exactly — Python's twin of
    ``Math.fround``."""
    rounded: float = struct.unpack("<f", struct.pack("<f", value))[0]
    return rounded


def f32(value: float) -> float:
    """The ``f32`` format: the shortest decimal that parses back to ``value``'s float32. ``value``
    must already BE a float32 (else the format would not be lossless)."""
    if not math.isfinite(value) or fround(value) != value:
        raise BundleError(f"{value!r} is not a finite float32 value")
    for digits in range(1, 10):
        candidate = float(f"{value:.{digits}g}")
        if fround(candidate) == value:
            return candidate
    return value  # unreachable: 9 significant digits always round-trip a float32


def encode_f32(values: Sequence[float]) -> str:
    """The ``f32-b64`` format (see the module docstring). Every value must already BE a float32."""
    for value in values:
        if not math.isfinite(value) or fround(value) != value:
            raise BundleError(f"{value!r} is not a finite float32 value")
    raw = struct.pack(f"<{len(values)}f", *values)
    planes = b"".join(raw[plane::4] for plane in range(4))
    return base64.b64encode(planes).decode("ascii")


def decode_f32(text: str, n: int) -> tuple[float, ...]:
    """Inverse of :func:`encode_f32` for ``n`` values (the page's decoder, in Python)."""
    try:
        planes = base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise BundleError(f"not f32-b64: {exc}") from exc
    if len(planes) != 4 * n:
        raise BundleError(f"f32-b64 holds {len(planes)} bytes, expected {4 * n}")
    raw = bytes(planes[plane * n + i] for i in range(n) for plane in range(4))
    return tuple(struct.unpack(f"<{n}f", raw))


def sig(value: float, digits: int) -> float:
    """Round to ``digits`` significant digits (the ``sig11`` / ``sig6`` formats)."""
    if not math.isfinite(value):
        raise BundleError(f"non-finite value {value!r} cannot enter the bundle")
    return float(f"{value:.{digits}g}")


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Max-subtracted softmax, summed left to right (the mirror's arithmetic)."""
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = 0.0
    for value in exps:
        total += value
    return tuple(value / total for value in exps)


# --- the stdlib replay (the env's lag-0 contract, mirrored) ---------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """One exhibit decision at bar ``index``: the float32 observation, the policy's logits,
    the requested (argmax) action, the effective action after the legality rule, and the position
    HELD at decision time (fills at closes ``<= index``)."""

    index: int
    observation: tuple[float, ...]
    logits: tuple[float, ...]
    requested: RlAction
    effective: RlAction
    position: float


@dataclass(frozen=True, slots=True)
class FoldReplay:
    fold_index: int
    decisions: tuple[Decision, ...]


def _require_env_actions(policy: MlpPolicy) -> None:
    if policy.actions != ENV_ACTION_ORDER:
        raise BundleError(
            f"policy {policy.policy_id} maps actions {policy.actions}, not the env's "
            f"{ENV_ACTION_ORDER}"
        )


def _legal(requested: RlAction, position: float) -> RlAction:
    if (requested is RlAction.BUY and position == 0.0) or (
        requested is RlAction.SELL and position == 1.0
    ):
        return requested
    return RlAction.HOLD


def replay_fold(closes: Sequence[float], fold: WalkForwardFold, policy: MlpPolicy) -> FoldReplay:
    """Replay ``policy`` over ``fold.test`` with the stdlib forward pass (what the page runs):
    flat start, observation from ``closes[: t + 1]`` + the held position (entry = the FILL close),
    float32-rounded, argmax, legality, fill at ``t + 1``. Cost-free: actions never depend on cost.
    """
    _require_env_actions(policy)
    lo, hi = fold.test
    if lo < MAX_LOOKBACK or hi > len(closes) - 1:
        raise BundleError(f"fold {fold.index}'s test window {fold.test} does not fit the bars")
    position = 0.0
    entry: int | None = None
    decisions: list[Decision] = []
    for t in range(lo, hi):
        values = build_observation(
            closes[: t + 1],
            held=position == 1.0,
            entry_close=None if entry is None else closes[entry],
            spec=policy.features,
        )
        observation = tuple(fround(value) for value in values)
        logits = policy.logits(observation)
        requested = policy.actions[policy.action_index(observation)]
        effective = _legal(requested, position)
        decisions.append(
            Decision(
                index=t,
                observation=observation,
                logits=logits,
                requested=requested,
                effective=effective,
                position=position,
            )
        )
        if effective is RlAction.BUY:
            position, entry = 1.0, t + 1
        elif effective is RlAction.SELL:
            position, entry = 0.0, None
    return FoldReplay(fold_index=fold.index, decisions=tuple(decisions))


def equity_path(
    closes: Sequence[float], start: int, actions: Sequence[RlAction], *, one_way_bps: float
) -> tuple[float, ...]:
    """Equity marks ``E_0 = 1.0 .. E_n`` of a flat-start episode deciding ``actions`` at bars
    ``start .. start + n - 1`` (requested actions; the legality rule is applied here), with the
    env's float arithmetic (see the module docstring)."""
    rate = one_way_bps * BPS
    equity, position = 1.0, 0.0
    marks = [equity]
    for offset, requested in enumerate(actions):
        t = start + offset
        move = 1.0 + position * (closes[t + 1] / closes[t] - 1.0)
        effective = _legal(requested, position)
        after = position
        if effective is RlAction.BUY:
            after = 1.0
        elif effective is RlAction.SELL:
            after = 0.0
        equity *= move * (1.0 - rate * abs(after - position))
        position = after
        marks.append(equity)
    return tuple(marks)


# --- stitching ------------------------------------------------------------------------------------


def stitch(per_fold: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Chain-link per-fold equity marks (each starting at 1.0) into one curve: fold *k*'s marks
    are scaled by the product of the earlier folds' final equities, and its first mark (== the
    previous fold's last) is not repeated."""
    out: list[float] = []
    base = 1.0
    for k, marks in enumerate(per_fold):
        if not marks or marks[0] != 1.0:
            raise BundleError(f"fold episode {k} must start at equity 1.0")
        out.extend(base * value for value in (marks if k == 0 else marks[1:]))
        base *= marks[-1]
    return tuple(out)


def quantile_band(curves: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Pointwise p25 / p50 / p75 over >= 2 equal-length curves (``statistics.quantiles`` with
    ``n=20, method="inclusive"``, cuts 4 / 9 / 14 — ``rl.metrics.null_band``'s method)."""
    if len(curves) < 2 or len({len(curve) for curve in curves}) != 1:
        raise BundleError("a band needs >= 2 curves of equal length")
    bands: list[list[float]] = [[], [], []]
    for i in range(len(curves[0])):
        cuts = statistics.quantiles([curve[i] for curve in curves], n=20, method="inclusive")
        for band, cut in zip(bands, (cuts[4], cuts[9], cuts[14]), strict=True):
            band.append(cut)
    return tuple(tuple(band) for band in bands)


# --- inputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExhibitPolicy:
    """One fold's exhibit-seed policy (already identity-checked by the job) and its provenance."""

    fold_index: int
    policy: MlpPolicy
    artifact_key: str
    artifact_sha256: str
    run_id: str
    training_library_versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CostCurves:
    """Per-fold env equity marks (fold order) at one cost: the exhibit agent, EVERY agent seed
    (the envelope), the three deterministic baselines, and per random seed the uniform and the
    turnover-matched random baselines (rl-10's ``random`` / ``random_matched[ppo]`` replays)."""

    cost_bps: float
    agent: tuple[tuple[float, ...], ...]
    agent_seeds: Mapping[int, tuple[tuple[float, ...], ...]]
    baselines: Mapping[str, tuple[tuple[float, ...], ...]]
    uniform: Mapping[int, tuple[tuple[float, ...], ...]]
    matched: Mapping[int, tuple[tuple[float, ...], ...]]


@dataclass(frozen=True, slots=True)
class BundleInputs:
    dataset_header: Mapping[str, Any]  # the raw spy_daily.json header, verbatim
    dataset_sha256: str
    session_dates: tuple[str, ...]  # per bar, ISO dates
    closes: tuple[float, ...]
    folds: tuple[WalkForwardFold, ...]
    report: Mapping[str, Any]  # the rl-10 report JSON, verbatim
    report_sha256: str
    exhibit_seed: int
    agent_seeds: tuple[int, ...]
    random_seeds: tuple[int, ...]
    policies: tuple[ExhibitPolicy, ...]
    replays: tuple[FoldReplay, ...]
    curves: tuple[CostCurves, ...]
    generated_at: str
    generated_at_source: str
    git: Mapping[str, Any]
    library_versions: Mapping[str, str]
    holdout_expected: bool  # some test window lies in the §5.6 final look-once holdout
    agent_id: str = "ppo"


# --- sections -------------------------------------------------------------------------------------


def _span(dates: Sequence[str], window: Sequence[int]) -> dict[str, Any]:
    lo, hi = int(window[0]), int(window[1])
    return {
        "bars": [lo, hi],
        "first_session": dates[lo] if hi > lo else None,
        "last_session": dates[hi - 1] if hi > lo else None,
    }


def _fold_meta(dates: Sequence[str], fold: WalkForwardFold) -> dict[str, Any]:
    return {
        "index": fold.index,
        "purge_bars": fold.purge_bars,
        "train": _span(dates, fold.train),
        "val": _span(dates, fold.val),
        "purge": _span(dates, fold.purge),
        "test": _span(dates, fold.test),
        "test_last_mark_session": dates[fold.test[1]],
    }


def _layer(layer: DenseLayer) -> dict[str, Any]:
    weights = [v for row in layer.weight for v in row]
    weight, bias = encode_f32(weights), encode_f32(layer.bias)
    if decode_f32(weight, len(weights)) != tuple(weights) or decode_f32(
        bias, len(layer.bias)
    ) != tuple(layer.bias):
        raise BundleError("f32-b64 did not round-trip a layer exactly")
    return {
        "activation": layer.activation,
        "in": layer.in_features,
        "out": layer.out_features,
        "bias": bias,
        "weight": weight,
    }


def layer_from_bundle(raw: Mapping[str, Any]) -> DenseLayer:
    """Decode one bundled layer back into the exact :class:`DenseLayer` (the page's decode path,
    in Python; tests rebuild every shipped policy with it)."""
    n_in, n_out = int(raw["in"]), int(raw["out"])
    flat = decode_f32(str(raw["weight"]), n_in * n_out)
    return DenseLayer(
        weight=tuple(tuple(flat[i * n_in : (i + 1) * n_in]) for i in range(n_out)),
        bias=decode_f32(str(raw["bias"]), n_out),
        activation=raw["activation"],
    )


def _architecture(policy: MlpPolicy) -> dict[str, Any]:
    return {
        "obs_dim": policy.obs_dim,
        "hidden": [layer.out_features for layer in policy.hidden],
        "hidden_activations": [layer.activation for layer in policy.hidden],
        "n_actions": policy.action_head.out_features,
        "actions": [action.value for action in policy.actions],
        "forward": _POLICY_FORWARD,
    }


def _parity_vector(fold_index: int, policy: MlpPolicy, obs: Sequence[float]) -> dict[str, Any]:
    observation = tuple(fround(v) for v in obs)
    logits = policy.logits(observation)
    return {
        "fold_index": fold_index,
        "observation": [f32(v) for v in observation],
        "logits": list(logits),
        "probs": list(softmax(logits)),
        "argmax": policy.action_index(observation),
    }


def parity_vectors(
    policies: Sequence[ExhibitPolicy], replays: Sequence[FoldReplay]
) -> list[dict[str, Any]]:
    """>= 1 vector per shipped policy (>= :data:`MIN_PARITY_VECTORS` overall): from each fold's
    real replay its first test decision, its first HELD decision (if any), then evenly spaced
    decisions until ``max(2, ceil(MIN_PARITY_VECTORS / n_policies))`` per policy; plus the
    saturated corners (all +10 / all -10, flat and held) through the first policy."""
    by_fold = {replay.fold_index: replay for replay in replays}
    per_policy = max(2, math.ceil(MIN_PARITY_VECTORS / len(policies)))
    vectors: list[dict[str, Any]] = []
    for item in policies:
        decisions = by_fold[item.fold_index].decisions
        held = next((i for i, d in enumerate(decisions) if d.position == 1.0), None)
        picks = [0] + ([held] if held is not None else [])
        n = len(decisions)
        for step in range(per_policy):
            if len(picks) >= min(per_policy, n):
                break
            candidate = (step * (n - 1)) // max(1, per_policy - 1)
            if candidate not in picks:
                picks.append(candidate)
        for index in picks:
            vectors.append(
                _parity_vector(item.fold_index, item.policy, decisions[index].observation)
            )
    first = policies[0]
    for value in (OBS_HIGH, OBS_LOW):
        for flag in (0.0, 1.0):
            corner = [value] * (first.policy.obs_dim - 2) + [flag, value if flag else 0.0]
            vectors.append(_parity_vector(first.fold_index, first.policy, corner))
    if len(vectors) < MIN_PARITY_VECTORS:
        raise BundleError(f"only {len(vectors)} parity vectors (need >= {MIN_PARITY_VECTORS})")
    return vectors


def _features(policy: MlpPolicy) -> dict[str, Any]:
    spec = policy.features.to_dict()
    names = list(policy.features.names)
    return {
        **spec,
        "obs_low": OBS_LOW,
        "obs_high": OBS_HIGH,
        "observation_dtype": "float32",
        "lookbacks": list(FEATURE_LOOKBACKS),
        "definitions": [
            {"name": name, "formula": formula, "lookback": lookback}
            for name, formula, lookback in zip(
                names, _FEATURE_FORMULAS, FEATURE_LOOKBACKS, strict=True
            )
        ],
        "arithmetic": list(_ARITHMETIC),
        "first_decision_index": MAX_LOOKBACK,
        "source": "trading.domain.rl_features (task-rl-04): THE shared observation builder",
    }


def _cost_key(cost: float) -> str:
    return f"{cost:g}"


def _envelope(curves: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pointwise min / max over equal-length curves (the seed envelope's p0 / p100)."""
    if len(curves) < 2 or len({len(curve) for curve in curves}) != 1:
        raise BundleError("an envelope needs >= 2 curves of equal length")
    columns = list(zip(*curves, strict=True))
    return tuple(min(c) for c in columns), tuple(max(c) for c in columns)


def _curves(inputs: BundleInputs, *, include_uniform: bool) -> dict[str, Any]:
    folds = inputs.folds
    dates = inputs.session_dates
    first, last = folds[0].test[0], folds[-1].test[1]
    segments = []
    position = 0
    for k, fold in enumerate(folds):
        if k > 0 and fold.test[0] != folds[k - 1].test[1]:
            raise BundleError(
                f"fold {fold.index}'s test window does not start where fold "
                f"{folds[k - 1].index}'s ends: the stitched curve would have a gap"
            )
        length = fold.test[1] - fold.test[0]
        segments.append(
            {
                "fold_index": fold.index,
                "start": position,
                "end": position + length,
                "first_session": dates[fold.test[0]],
                "last_mark_session": dates[fold.test[1]],
            }
        )
        position += length
    n_seeds = len(inputs.agent_seeds)
    n_random = len(inputs.random_seeds)
    by_cost: dict[str, Any] = {}
    exhibit_on_top: list[bool] = []
    for curves in inputs.curves:
        exact: dict[str, tuple[float, ...]] = {AGENT_SERIES: stitch(curves.agent)}
        for name in BASELINE_SERIES:
            exact[name] = stitch(curves.baselines[name])
        display: dict[str, tuple[float, ...]] = {}
        seed_curves = [stitch(curves.agent_seeds[s]) for s in inputs.agent_seeds]
        if stitch(curves.agent_seeds[inputs.exhibit_seed]) != exact[AGENT_SERIES]:
            raise BundleError("the exhibit curve is not its own seed's replay")
        display.update(zip(ENVELOPE_SERIES, _envelope(seed_curves), strict=True))
        exhibit_on_top.append(exact[AGENT_SERIES][-1] == display[ENVELOPE_SERIES[1]][-1])
        matched = [stitch(curves.matched[s]) for s in inputs.random_seeds]
        display.update(zip(MATCHED_BAND_SERIES, quantile_band(matched), strict=True))
        if include_uniform:
            uniform = [stitch(curves.uniform[s]) for s in inputs.random_seeds]
            display.update(zip(UNIFORM_BAND_SERIES, quantile_band(uniform), strict=True))
        for name, values in (*exact.items(), *display.items()):
            if len(values) != last - first + 1:
                raise BundleError(f"{name} at {curves.cost_bps:g} bps has {len(values)} marks")
        by_cost[_cost_key(curves.cost_bps)] = {
            **{name: [sig(v, CURVE_DIGITS) for v in values] for name, values in exact.items()},
            **{name: [sig(v, BAND_DIGITS) for v in values] for name, values in display.items()},
        }
    background = [
        {
            "fold_index": fold.index,
            "kind": kind,
            "first_session": dates[window[0]],
            "last_session": dates[window[1] - 1],
        }
        for fold in folds
        for kind, window in (("train", fold.train), ("val", fold.val), ("purge", fold.purge))
        if window[1] > window[0]
    ]
    top = (
        "the exhibit line ends at the TOP of this envelope at every cost (it is the best final "
        "equity of the seeds)"
        if all(exhibit_on_top)
        else "the exhibit line does not end at the top of this envelope at every cost"
    )
    series = {
        AGENT_SERIES: f"exhibit seed {inputs.exhibit_seed}'s per-fold PPO policies (ONE seed; "
        "see meta.exhibit_rank)",
        ENVELOPE_SERIES[0]: f"pointwise minimum over all {n_seeds} agent seeds' stitched curves "
        "(the shaded 5-seed envelope's floor)",
        ENVELOPE_SERIES[1]: f"pointwise maximum over all {n_seeds} agent seeds' stitched curves; "
        + top,
        "buy_and_hold": "BUY at each fold's first decision, then HOLD",
        "always_flat": "never trades (equity 1.0)",
        "sma_cross_20_50": "long iff SMA_20 > SMA_50 of the trailing closes",
        MATCHED_BAND_SERIES[0]: f"{MATCHED_BAND_LABEL}: pointwise 25th percentile over "
        f"{n_random} chain-linked seeds; in each fold the random policy trades with the agent's "
        "own per-fold trade rate (rl-07 matched_trade_probability over its "
        f"{n_seeds} seeds), seed s = random.Random(s) — exactly rl-10's random_matched[ppo]",
        MATCHED_BAND_SERIES[1]: f"{MATCHED_BAND_LABEL}: pointwise median",
        MATCHED_BAND_SERIES[2]: f"{MATCHED_BAND_LABEL}: pointwise 75th percentile",
    }
    if include_uniform:
        series.update(
            {
                UNIFORM_BAND_SERIES[0]: f"{UNIFORM_BAND_LABEL}: pointwise 25th percentile over "
                f"{n_random} chain-linked uniform (1/3 trade probability) seeds — rl-10's random",
                UNIFORM_BAND_SERIES[1]: f"{UNIFORM_BAND_LABEL}: pointwise median",
                UNIFORM_BAND_SERIES[2]: f"{UNIFORM_BAND_LABEL}: pointwise 75th percentile",
            }
        )
    formats = dict.fromkeys((AGENT_SERIES, *BASELINE_SERIES), f"sig{CURVE_DIGITS}")
    display_names = (
        *ENVELOPE_SERIES,
        *MATCHED_BAND_SERIES,
        *(UNIFORM_BAND_SERIES if include_uniform else ()),
    )
    formats.update(dict.fromkeys(display_names, f"sig{BAND_DIGITS}"))
    return {
        "dates": list(dates[first : last + 1]),
        "segments": segments,
        "by_cost": by_cost,
        "background": background,
        "series": series,
        "headline_band": "random_matched",
        "bands": {
            "random_matched": {"label": MATCHED_BAND_LABEL, "gate_learn_null": True},
            **(
                {"random_uniform": {"label": UNIFORM_BAND_LABEL, "gate_learn_null": False}}
                if include_uniform
                else {}
            ),
        },
        "dropped": []
        if include_uniform
        else [
            "random_uniform band: dropped to fit the size budget (the matched band is the honest "
            "default)"
        ],
        "semantics": (
            "OUT-OF-SAMPLE TEST SEGMENTS ONLY: dates[i] is the session whose close marks point i; "
            "point 0 is the first test decision (equity 1.0). Each fold's episode starts flat at "
            "equity 1.0 and is chain-linked onto the previous fold's final equity (no cost at the "
            "boundary; the next fold pays its own entry). The HEADLINE band (headline_band) is "
            f"random_matched_* — {MATCHED_BAND_LABEL}: the null GATE-LEARN failed to beat; the "
            "agent line beating a uniform-random band is NOT evidence of learning. "
            "agent_seeds_min/max shade all agent seeds; the agent line is ONE of them. "
            "Train/val/purge appear ONLY as 'background' spans (no values)"
        ),
        "formats": formats,
    }


def _replay(inputs: BundleInputs) -> dict[str, Any]:
    dates = inputs.session_dates
    records = [
        {
            "date": dates[decision.index],
            "fold_index": replay.fold_index,
            "action": decision.requested.value,
            "effective": decision.effective.value,
            "action_probs": [sig(p, PROB_DIGITS) for p in softmax(decision.logits)],
            "position": decision.position,
        }
        for replay in inputs.replays
        for decision in replay.decisions
    ]
    return {
        "exhibit_seed": inputs.exhibit_seed,
        "records": records,
        "semantics": (
            "one record per TEST decision bar: action = the policy's argmax (requested), effective "
            "= after the legality rule (BUY only when flat, SELL only when long; else a HOLD "
            "no-op), position = held at decision time (fills at closes <= date). An accepted "
            "order fills at the NEXT session's close. fold_index names the policy (policy.policies"
            "[].fold_index) that decided it; each fold starts flat"
        ),
    }


def exhibit_rank(
    report: Mapping[str, Any],
    *,
    agent_id: str,
    seeds: Sequence[int],
    folds: Sequence[WalkForwardFold],
    exhibit_seed: int,
) -> dict[str, Any]:
    """Where the exhibit seed stands among ALL the agent's seeds, per cost, from the report's own
    rows (no replay): each seed's stitched final equity (the product of its per-fold final
    equities, fold order — what its stitched curve ends at) and its mean per-fold sharpe. Rank 1 =
    best."""
    rows = {
        (int(r["seed"]), int(r["fold_index"]), float(r["cost_bps"])): r
        for r in report["results"]
        if r["policy_id"] == agent_id
    }
    per_cost: dict[str, Any] = {}
    for cost in (float(c) for c in report["cost_bps_grid"]):
        entries = []
        for seed in seeds:
            try:
                fold_rows = [rows[(seed, fold.index, cost)] for fold in folds]
            except KeyError as exc:
                raise BundleError(f"the report has no {agent_id} row for {exc}") from exc
            final = 1.0
            for row in fold_rows:
                final *= float(row["final_equity"])
            entries.append(
                {
                    "seed": seed,
                    "stitched_final_equity": final,
                    "mean_fold_sharpe": math.fsum(float(r["sharpe"]) for r in fold_rows)
                    / len(fold_rows),
                }
            )
        mine = next(e for e in entries if e["seed"] == exhibit_seed)
        per_cost[_cost_key(cost)] = {
            "exhibit_rank_by_final_equity": 1
            + sum(e["stitched_final_equity"] > mine["stitched_final_equity"] for e in entries),
            "exhibit_rank_by_mean_fold_sharpe": 1
            + sum(e["mean_fold_sharpe"] > mine["mean_fold_sharpe"] for e in entries),
            "seeds": entries,
        }
    best = all(v["exhibit_rank_by_final_equity"] == 1 for v in per_cost.values())
    return {
        "exhibit_seed": exhibit_seed,
        "n_seeds": len(seeds),
        "best_by_final_equity_at_every_cost": best,
        "by": "stitched final equity = product of the seed's per-fold final_equity rows; mean "
        "per-fold sharpe; both from the evaluation report's rows (rank 1 = best)",
        "per_cost": per_cost,
    }


def _standing(rank: Mapping[str, Any]) -> str:
    n = rank["n_seeds"]
    if rank["best_by_final_equity_at_every_cost"]:
        return (
            f"It is the best of {n} seeds on this span (rank 1 of {n} by stitched final equity at "
            "every cost; meta.exhibit_rank), so it flatters the agent."
        )
    ranks = ", ".join(
        f"{v['exhibit_rank_by_final_equity']} of {n} at {cost} bps"
        for cost, v in rank["per_cost"].items()
    )
    return f"By stitched final equity it ranks {ranks} on this span (meta.exhibit_rank)."


def _verdict_word(gate: Mapping[str, Any]) -> str:
    """rl-10's printed vocabulary (``jobs/rl_eval._verdict_word``)."""
    passed = gate["passed"]
    if passed is None:
        return "VOID"
    if gate["name"] == "GATE-OVERFIT":
        return "OK" if passed else "WARN"
    if gate["name"] == VS_BUY_AND_HOLD:
        return "YES" if passed else "NO"
    return "PASS" if passed else "FAIL"


def _sharpe_by_fold(
    results: Sequence[Mapping[str, Any]], policy_id: str, cost: float
) -> dict[str, list[float]]:
    """``rl_evaluation._by_fold``: rows sorted by ``(fold, seed)``, grouped by ``str(fold)``."""
    rows = [r for r in results if r["policy_id"] == policy_id and r["cost_bps"] == cost]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (int(r["fold_index"]), int(r["seed"]))):
        grouped[str(row["fold_index"])].append(float(row["sharpe"]))
    return dict(grouped)


def _note(notes: Sequence[str], needle: str) -> str | None:
    return next((note for note in notes if needle in note), None)


def eval_table(
    report: Mapping[str, Any], *, report_sha256: str, holdout_expected: bool
) -> dict[str, Any]:
    """Look #N's numbers, transformed — copied verbatim wherever they exist. The ONLY recompute is
    the sharpe IQM CI of the single-run baselines the report left at None (see
    :data:`CI_RECOMPUTE_NOTE`); every CI the report did record is re-derived from its own rows and
    must match exactly, proving the recompute is the report's estimator."""
    results = report["results"]
    rows: list[dict[str, Any]] = []
    for summary in report["summaries"]:
        row = dict(summary)
        ci = two_stage_bootstrap_iqm_ci(_sharpe_by_fold(results, row["policy_id"], row["cost_bps"]))
        recorded = (row["sharpe_iqm_ci_low"], row["sharpe_iqm_ci_high"])
        if recorded != (None, None):
            if ci != recorded:
                raise BundleError(
                    f"{row['policy_id']} at {row['cost_bps']} bps: the report's sharpe IQM CI "
                    f"{recorded} is not reproduced from its own rows ({ci}) — refusing to "
                    "recompute anything from this report"
                )
            row["sharpe_iqm_ci_source"] = "report"
        elif row["policy_id"] in CI_RECOMPUTABLE and ci is not None:
            row["sharpe_iqm_ci_low"], row["sharpe_iqm_ci_high"] = ci
            row["sharpe_iqm_ci_note"] = RECOMPUTED_CI_NOTE
            row["sharpe_iqm_ci_source"] = "recomputed"
        else:
            row["sharpe_iqm_ci_source"] = "unavailable"
        rows.append(row)
    gates = []
    verdicts: dict[str, str] = {}
    for gate in report["gates"]:
        item = dict(gate)
        item["verdict"] = _verdict_word(gate)
        gates.append(item)
        if gate["name"] == VS_BUY_AND_HOLD:
            verdicts[gate["policy_id"]] = buy_and_hold_line(GateVerdict(**gate))
    agents = [g["policy_id"] for g in report["gates"] if g["name"] == VS_BUY_AND_HOLD]
    if not agents:
        raise BundleError("the report has no vs BUY-AND-HOLD verdict")
    recomputed = any(row["sharpe_iqm_ci_source"] == "recomputed" for row in rows)
    replaced = recomputed and any(STALE_CI_NOTE_NEEDLE in note for note in report["notes"])
    notes = [
        CI_NOTE_REPLACEMENT if recomputed and STALE_CI_NOTE_NEEDLE in note else note
        for note in report["notes"]
    ]
    look = int(report["n_configurations_evaluated"])
    holdout = _note(notes, "look-once holdout")
    amendment = _note(notes, "AMENDED")
    if look > 1 and amendment is None:
        raise BundleError(
            f"look #{look} but the report carries no gate-rule amendment note — refusing to ship "
            "an honesty block without it"
        )
    if look > 1 and holdout_expected and holdout is None:
        raise BundleError(
            f"look #{look} with test windows inside the look-once holdout, but the report carries "
            "no holdout note — refusing to ship an honesty block without it"
        )
    return {
        "source": {
            "report_sha256": report_sha256,
            "as_of": report["as_of"],
            "look": report["n_configurations_evaluated"],
            "symbol": report["symbol"],
            "dataset_sha256": report["dataset_sha256"],
        },
        "agent": agents[0],
        "headline_verdict": verdicts[agents[0]],
        "beats_buy_and_hold": verdicts,
        "n_folds": report["n_folds"],
        "n_seeds": report["n_seeds"],
        "low_n": report["low_n"],
        "cost_bps_grid": list(report["cost_bps_grid"]),
        "gate_cost_bps": report["gate_cost_bps"],
        "break_even_cost_bps": dict(report["break_even_cost_bps"]),
        "edge_over_flat_log_growth": report["edge_over_flat_log_growth"],
        "rows": rows,
        "gates": gates,
        "parity": {
            "ties": report["parity_ties"],
            "observations": report["parity_observations"],
        },
        "evaluation_mode": report["evaluation_mode"],
        "caveats": {
            "look_count": report["n_configurations_evaluated"],
            "n_configurations_source": report["n_configurations_source"],
            "holdout": holdout,
            "gate_rule_amendment": amendment,
            "gate_rules_version": report["gate_rules_version"],
        },
        "warnings": list(report["warnings"]),
        "notes": notes,
        "bundle_notes": [CI_RECOMPUTE_NOTE]
        + (
            [
                "eval_table.notes are the report's notes verbatim EXCEPT its IQM-CI note, "
                "replaced (marked '[rl-12: ...]') because it predates the recomputed baseline CIs "
                "and would contradict the rows"
            ]
            if replaced
            else []
        ),
    }


# --- the bundle -----------------------------------------------------------------------------------


def assemble(inputs: BundleInputs, *, include_uniform: bool = True) -> dict[str, Any]:
    """The whole bundle as plain JSON types (serialize with :func:`bundle_to_json`).
    ``include_uniform=False`` drops the optional uniform-random band (the size-budget fallback)."""
    if not inputs.policies or len(inputs.policies) != len(inputs.folds):
        raise BundleError("exactly one exhibit policy per fold is required")
    if [p.fold_index for p in inputs.policies] != [f.index for f in inputs.folds]:
        raise BundleError("exhibit policies must be in fold order")
    if [r.fold_index for r in inputs.replays] != [f.index for f in inputs.folds]:
        raise BundleError("replays must be in fold order")
    header = inputs.dataset_header
    dates = inputs.session_dates
    report = inputs.report
    first_policy = inputs.policies[0].policy
    if any(p.policy.features != first_policy.features for p in inputs.policies):
        raise BundleError("every exhibit policy must share one feature spec")
    if any(_architecture(p.policy) != _architecture(first_policy) for p in inputs.policies):
        raise BundleError("every exhibit policy must share one architecture")
    generated = str(header["generated_at"])
    rank = exhibit_rank(
        report,
        agent_id=inputs.agent_id,
        seeds=inputs.agent_seeds,
        folds=inputs.folds,
        exhibit_seed=inputs.exhibit_seed,
    )
    meta = {
        "generated_at": inputs.generated_at,
        "generated_at_source": inputs.generated_at_source,
        "git_commit": inputs.git.get("git_commit"),
        "git_dirty": inputs.git.get("git_dirty"),
        "symbol": header["symbol"],
        "adjustment": header["adjustment"],
        "dataset_generated_at": header["generated_at"],
        "price_basis": header["price_basis"],
        "price_label": PRICE_LABEL_TEMPLATE.format(generated_at=generated),
        "dataset": {
            "sha256": inputs.dataset_sha256,
            "n_bars": len(dates),
            "first_session": dates[0],
            "last_session": dates[-1],
            "source_adapter": header["source_adapter"],
        },
        "eval_report_sha256": inputs.report_sha256,
        "exhibit_seed": inputs.exhibit_seed,
        "exhibit_label": EXHIBIT_LABEL_TEMPLATE.format(
            seed=inputs.exhibit_seed,
            standing=_standing(rank),
            n_seeds=report["n_seeds"],
            n_folds=report["n_folds"],
        ),
        "exhibit_rank": rank,
        "seeds": {
            "agent": list(inputs.agent_seeds),
            "exhibit": inputs.exhibit_seed,
            "random": list(inputs.random_seeds),
        },
        "cost_bps_grid": [c.cost_bps for c in inputs.curves],
        "folds": [_fold_meta(dates, fold) for fold in inputs.folds],
        "policy_sha256s": {p.artifact_key: p.artifact_sha256 for p in inputs.policies},
        "policies_shipped": {
            "count": len(inputs.policies),
            "of_folds": len(inputs.folds),
            "precision": "full: the exact float32 weights of every exhibit-seed fold policy "
            "(f32-b64, lossless)",
        },
        "library_versions": dict(inputs.library_versions),
        "execution": {
            "execution_lag": 0,
            "initial_equity": 1.0,
            "position_sizes": [0.0, 1.0],
            "fill": "an accepted BUY/SELL decided at bar t fills at the close of bar t+1",
            "equity_step": "E_{t+1} = E_t * ((1 + p_t * (C_{t+1} / C_t - 1)) * (1 - (bps * "
            "1e-4) * |p_{t+1} - p_t|)), p_t held over t->t+1, p_{t+1} after the fill at C_{t+1}",
            "legality": "BUY only when flat, SELL only when long; anything else is a HOLD no-op",
            "fold_reset": "each fold's episode starts flat at equity 1.0; stitched curves "
            "chain-link the per-fold episodes",
            "cost_model": "flat one-way bps of traded notional, charged on every fill "
            "(trading.rl.costs; scale-free, never a price level)",
        },
        "float_formats": dict(_FORMATS),
        "schema": SCHEMA_FILE,
    }
    test_ranges = {fold.index: _span(dates, fold.test) for fold in inputs.folds}
    train_ranges = {fold.index: _span(dates, fold.train) for fold in inputs.folds}
    policies = [
        {
            "fold_index": item.fold_index,
            "policy_id": item.policy.policy_id,
            "artifact_sha256": item.artifact_sha256,
            "run_id": item.run_id,
            "training_library_versions": dict(item.training_library_versions),
            "train": train_ranges[item.fold_index],
            "test": test_ranges[item.fold_index],
            "hidden": [_layer(layer) for layer in item.policy.hidden],
            "action_head": _layer(item.policy.action_head),
        }
        for item in inputs.policies
    ]
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "meta": meta,
        "bars": [
            {"date": date, "close": close} for date, close in zip(dates, inputs.closes, strict=True)
        ],
        "features": _features(first_policy),
        "policy": {
            "architecture": _architecture(first_policy),
            "weights_format": "f32-b64",
            "policies": policies,
            "parity_vectors": parity_vectors(inputs.policies, inputs.replays),
            "parity_semantics": (
                "observation (f32) -> logits/probs (exact float64, computed by trading.domain."
                "rl_policy from the exact weights) and argmax; the page's forward pass must "
                "match argmax exactly and logits within a tolerance (tanh/exp ~1 ulp)"
            ),
        },
        "replay": _replay(inputs),
        "curves": _curves(inputs, include_uniform=include_uniform),
        "eval_table": eval_table(
            report, report_sha256=inputs.report_sha256, holdout_expected=inputs.holdout_expected
        ),
    }


def bundle_to_json(bundle: Mapping[str, Any]) -> bytes:
    """The bundle's exact bytes (see the module docstring's writer)."""
    try:
        text = json.dumps(
            bundle, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
        )
    except (TypeError, ValueError) as exc:
        raise BundleError(f"the bundle is not strict JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def gzip_size(data: bytes) -> int:
    """Deterministic gzip size (level 9, mtime 0) — what the size budget is measured on."""
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def check_size(data: bytes) -> tuple[int, int]:
    """``(raw, gzip)`` bytes; raises :class:`BundleError` over :data:`MAX_RAW_BYTES` /
    :data:`MAX_GZIP_BYTES`."""
    raw, packed = len(data), gzip_size(data)
    if raw > MAX_RAW_BYTES or packed > MAX_GZIP_BYTES:
        raise BundleError(
            f"bundle is {raw} bytes raw / {packed} gzip; the budget is {MAX_RAW_BYTES} / "
            f"{MAX_GZIP_BYTES}"
        )
    return raw, packed


def secret_like(text: str) -> list[str]:
    """Every key-like / account-like / local-path-like match in ``text`` (empty = clean)."""
    return [m.group(0) for pattern in _SECRET_PATTERNS for m in pattern.finditer(text)]


def assert_curves_test_only(bundle: Mapping[str, Any]) -> None:
    """Fail unless every curve point is an out-of-sample TEST mark of the fold that governs it:
    the curve dates are exactly the stitched test marks (first fold's first test decision through
    the last fold's last mark), each segment lies inside its own fold's test marks and outside
    that fold's train/val/purge, and every series has one value per date."""
    folds = {fold["index"]: fold for fold in bundle["meta"]["folds"]}
    curves = bundle["curves"]
    dates: list[str] = curves["dates"]
    if dates != sorted(set(dates)):
        raise BundleError("curve dates must be strictly ascending")
    covered: set[str] = set()
    expected_start = 0
    for segment in curves["segments"]:
        fold = folds.get(segment["fold_index"])
        if fold is None:
            raise BundleError(f"curve segment names unknown fold {segment['fold_index']}")
        start, end = segment["start"], segment["end"]
        if start != expected_start or not 0 <= start < end < len(dates):
            raise BundleError(f"curve segment {segment} does not tile the curve")
        expected_start = end
        span = dates[start : end + 1]
        test = fold["test"]
        if (segment["first_session"], segment["last_mark_session"]) != (
            test["first_session"],
            fold["test_last_mark_session"],
        ):
            raise BundleError(
                f"curve segment {segment['fold_index']} labels {segment['first_session']}.."
                f"{segment['last_mark_session']}, not its fold's test marks"
            )
        if span[0] != test["first_session"] or span[-1] != fold["test_last_mark_session"]:
            raise BundleError(
                f"fold {fold['index']}'s curve segment {span[0]}..{span[-1]} is not its test "
                f"marks {test['first_session']}..{fold['test_last_mark_session']}"
            )
        for kind in ("train", "val", "purge"):
            window = fold[kind]
            if window["first_session"] is None:
                continue
            leaked = [d for d in span if window["first_session"] <= d <= window["last_session"]]
            if leaked:
                raise BundleError(
                    f"TRAIN-REGION LEAK: fold {fold['index']}'s curve covers its {kind} sessions "
                    f"{leaked[0]}..{leaked[-1]}"
                )
        covered.update(span)
    if expected_start != len(dates) - 1 or covered != set(dates):
        raise BundleError("curve dates outside every fold's test marks")
    expected_background = sorted(
        (fold["index"], kind, fold[kind]["first_session"], fold[kind]["last_session"])
        for fold in folds.values()
        for kind in ("train", "val", "purge")
        if fold[kind]["first_session"] is not None
    )
    background = sorted(
        (b["fold_index"], b["kind"], b["first_session"], b["last_session"])
        for b in curves["background"]
    )
    if background != expected_background:
        raise BundleError("curve background spans are not exactly the folds' train/val/purge")
    for cost, series in curves["by_cost"].items():
        for name, values in series.items():
            if len(values) != len(dates):
                raise BundleError(f"curves {cost}/{name}: {len(values)} values for {len(dates)}")
