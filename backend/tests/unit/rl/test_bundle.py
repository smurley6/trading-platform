"""task-rl-12: the website demo bundle (``rl/bundle.py`` + ``jobs/rl_bundle.build``).

The eight spec acceptance tests run against a bundle built from a complete SYNTHETIC layout
(``bundle_fixture``: real dataset writer, real policy store, an rl-10 report from the real replay +
statistics code). No ``rl`` marker: the bundle is torch-free; the env replays it cross-checks need
only gymnasium + numpy (``[dev]``).
"""

import json
import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.unit.rl.bundle_fixture import Layout, build_layout
from tests.unit.rl.jsonschema_subset import SchemaViolationError, UnsupportedKeywordError, Validator
from trading.application.rl_evaluation import BEATS_BH_NO_CI_INCLUDES_ZERO
from trading.domain.rl_features import WARMUP_BARS, build_observation
from trading.domain.rl_policy import ENV_ACTION_ORDER, MlpPolicy
from trading.jobs import rl_bundle as job
from trading.rl import bundle as rl_bundle

SCHEMA = Path(__file__).parents[4] / "docs" / "demo-bundle.schema.json"


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> Layout:
    return build_layout(tmp_path_factory.mktemp("bundle") / "layout")


@pytest.fixture(scope="module")
def built(layout: Layout) -> tuple[bytes, dict[str, Any]]:
    return job.build(job.parse_args(layout.argv(layout.root / "demo.json")))


@pytest.fixture(scope="module")
def bundle(built: tuple[bytes, dict[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(built[0])
    return parsed


def _policies(bundle: dict[str, Any]) -> Iterator[tuple[int, MlpPolicy]]:
    from trading.rl.env import DEFAULT_FEATURE_SPEC

    for item in bundle["policy"]["policies"]:
        yield (
            item["fold_index"],
            MlpPolicy(
                policy_id=item["policy_id"],
                schema_version=1,
                features=DEFAULT_FEATURE_SPEC,
                hidden=tuple(rl_bundle.layer_from_bundle(raw) for raw in item["hidden"]),
                action_head=rl_bundle.layer_from_bundle(item["action_head"]),
                actions=ENV_ACTION_ORDER,
            ),
        )


def _fround(value: float) -> float:
    rounded: float = struct.unpack("<f", struct.pack("<f", value))[0]
    return rounded


# --- 1. schema ------------------------------------------------------------------------------------


def test_bundle_validates_against_the_checked_in_schema(bundle: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Validator(schema).validate(bundle)


def test_the_schema_check_has_teeth(bundle: dict[str, Any]) -> None:
    validator = Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    extra = json.loads(json.dumps(bundle))
    extra["meta"]["account_id"] = "x"
    with pytest.raises(SchemaViolationError, match="unexpected property"):
        validator.validate(extra)
    wrong = json.loads(json.dumps(bundle))
    wrong["replay"]["records"][0]["action"] = "short"
    with pytest.raises(SchemaViolationError, match="not in enum"):
        validator.validate(wrong)
    missing = json.loads(json.dumps(bundle))
    del missing["eval_table"]["headline_verdict"]
    with pytest.raises(SchemaViolationError, match="missing required"):
        validator.validate(missing)
    with pytest.raises(UnsupportedKeywordError):
        Validator({"type": "object", "patternProperties": {}}).validate({})


def test_schema_documents_the_price_label_guardrail() -> None:
    meta = json.loads(SCHEMA.read_text(encoding="utf-8"))["$defs"]["meta"]["properties"]
    label = "dividend-adjusted (total-return) closes as served by Alpaca on"
    for field in ("adjustment", "dataset_generated_at"):
        assert "PRICE-LABEL GUARDRAIL" in meta[field]["description"]
    assert label in meta["adjustment"]["description"]
    assert 'never as "SPY\'s price"' in meta["adjustment"]["description"]


# --- 2. parity vectors ----------------------------------------------------------------------------


def test_parity_vectors_match_the_stdlib_forward_pass(bundle: dict[str, Any]) -> None:
    policies = dict(_policies(bundle))
    vectors = bundle["policy"]["parity_vectors"]
    assert len(vectors) >= 20
    assert {v["fold_index"] for v in vectors} == set(policies)  # >= 1 per shipped policy
    for vector in vectors:
        policy = policies[vector["fold_index"]]
        observation = [_fround(v) for v in vector["observation"]]  # f32: apply Math.fround
        logits = policy.logits(observation)
        assert list(logits) == vector["logits"]
        assert policy.action_index(observation) == vector["argmax"]
        assert vector["probs"] == pytest.approx(list(rl_bundle.softmax(logits)), abs=0.0)
        assert math.fsum(vector["probs"]) == pytest.approx(1.0, abs=1e-12)


def test_shipped_weights_are_the_exact_evaluated_float32_weights(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    from trading.infrastructure.rl.policy_store import load_policy

    shipped = dict(_policies(bundle))
    assert len(shipped) == layout.n_folds == bundle["meta"]["policies_shipped"]["count"]
    for fold, policy in shipped.items():
        original = load_policy(
            layout.walk_forward / f"fold{fold:02d}" / "spy_ppo_seed0.policy.json"
        )
        assert policy.hidden == original.hidden and policy.action_head == original.action_head


# --- 3. client-recompute parity -------------------------------------------------------------------


def _client_curve(bundle: dict[str, Any], cost_bps: float) -> list[float]:
    """The page's recompute path, written independently of rl/bundle.py: (bars, replay actions,
    cost model) -> stitched agent equity."""
    closes = [bar["close"] for bar in bundle["bars"]]
    index = {bar["date"]: i for i, bar in enumerate(bundle["bars"])}
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for record in bundle["replay"]["records"]:
        by_fold.setdefault(record["fold_index"], []).append(record)
    curve: list[float] = []
    base = 1.0
    for k, fold in enumerate(sorted(by_fold)):
        equity, position = 1.0, 0.0
        marks = [1.0]
        for record in by_fold[fold]:
            t = index[record["date"]]
            target = position
            if record["action"] == "buy" and position == 0.0:
                target = 1.0
            elif record["action"] == "sell" and position == 1.0:
                target = 0.0
            move = 1.0 + position * (closes[t + 1] / closes[t] - 1.0)
            equity *= move * (1.0 - cost_bps * 1e-4 * abs(target - position))
            position = target
            marks.append(equity)
        curve.extend(base * m for m in (marks if k == 0 else marks[1:]))
        base *= marks[-1]
    return curve


def test_client_recompute_reproduces_the_bundled_agent_curve(bundle: dict[str, Any]) -> None:
    for cost_key, series in bundle["curves"]["by_cost"].items():
        client = _client_curve(bundle, float(cost_key))
        bundled = series["agent"]
        assert len(client) == len(bundled) == len(bundle["curves"]["dates"])
        for mine, theirs in zip(client, bundled, strict=True):
            assert abs(mine - theirs) <= 1e-9 * abs(theirs)


def test_client_rebuilds_every_replay_action_from_bars_and_weights(bundle: dict[str, Any]) -> None:
    from trading.rl.env import DEFAULT_FEATURE_SPEC

    policies = dict(_policies(bundle))
    closes = [bar["close"] for bar in bundle["bars"]]
    index = {bar["date"]: i for i, bar in enumerate(bundle["bars"])}
    position, entry, fold = 0.0, None, -1
    for record in bundle["replay"]["records"]:
        if record["fold_index"] != fold:
            position, entry, fold = 0.0, None, int(record["fold_index"])
        t = index[record["date"]]
        assert record["position"] == position
        obs = build_observation(
            closes[: t + 1],
            held=position == 1.0,
            entry_close=None if entry is None else closes[entry],
            spec=DEFAULT_FEATURE_SPEC,
        )
        policy = policies[fold]
        observation = [_fround(v) for v in obs]
        assert policy.act(observation).value == record["action"]
        probs = rl_bundle.softmax(policy.logits(observation))
        assert record["action_probs"] == pytest.approx(list(probs), rel=1e-5)
        if record["effective"] == "buy":
            position, entry = 1.0, t + 1
        elif record["effective"] == "sell":
            position, entry = 0.0, None


# --- 4. determinism -------------------------------------------------------------------------------


def test_two_runs_produce_byte_identical_bundles(
    layout: Layout, built: tuple[bytes, dict[str, Any]]
) -> None:
    again, _ = job.build(job.parse_args(layout.argv(layout.root / "again.json")))
    assert again == built[0]
    assert built[0].endswith(b"}\n") and built[0].count(b"\n") == 1
    assert rl_bundle.bundle_to_json(json.loads(built[0])) == built[0]  # sorted keys, fixed floats


def test_float_formats_are_the_documented_ones(bundle: dict[str, Any]) -> None:
    formats = bundle["curves"]["formats"]
    assert formats["agent"] == formats["buy_and_hold"] == "sig11"
    assert formats["random_matched_p50"] == formats["agent_seeds_max"] == "sig6"
    for series in bundle["curves"]["by_cost"].values():
        for name, values in series.items():
            digits = int(formats[name].removeprefix("sig"))
            assert all(float(f"{v:.{digits}g}") == v for v in values)
    for record in bundle["replay"]["records"]:
        assert all(float(f"{p:.6g}") == p for p in record["action_probs"])
    for vector in bundle["policy"]["parity_vectors"]:
        # the shortest decimal of the float32 it rounds to (lossless after Math.fround)
        assert all(rl_bundle.f32(_fround(v)) == v for v in vector["observation"])


# --- 5. hygiene -----------------------------------------------------------------------------------


def test_the_bundle_contains_no_key_like_strings_or_local_paths(
    layout: Layout, built: tuple[bytes, dict[str, Any]]
) -> None:
    text = built[0].decode("utf-8")
    assert rl_bundle.secret_like(text) == []
    for literal in ("APCA", "Bearer", "postgresql://", "postgres://", "/Users/", "/home/"):
        assert literal not in text
    assert str(layout.root) not in text and str(layout.root.resolve()) not in text


@pytest.mark.parametrize(
    "planted",
    [
        "APCA-API-KEY-ID",
        "APCA_API_SECRET_KEY=abc",
        "sk-ant-api03-abcdefgh",
        "Authorization: Bearer abc.def",
        "postgresql://user:pw@db.example.com:5432/x",
        "postgresql+asyncpg://u@prod-host/x",
        "PKABCDEFGHIJKLMNOP12",
        "/Users/someone/trading-platform/backend",
        "C:\\Users\\someone",
        "contact steven@example.com",
        "https://user:token@github.com/x.git",
        "/opt/homebrew/bin",
        "/Volumes/External/data",
        "/root/.ssh",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "API_KEY=abc",
    ],
)
def test_the_hygiene_scan_catches_planted_secrets(planted: str) -> None:
    assert rl_bundle.secret_like(json.dumps({"note": planted})) != []


def test_the_hygiene_scan_allows_local_postgres_and_ordinary_text() -> None:
    clean = "postgresql://localhost:5432/x task-rl-04 risk-free; tasks-like words; Bearer-less"
    assert rl_bundle.secret_like(clean) == []


# --- 6. size budget -------------------------------------------------------------------------------


def test_size_budget_is_asserted(
    built: tuple[bytes, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, packed = rl_bundle.check_size(built[0])
    assert raw == len(built[0]) <= 2_500_000
    assert packed == rl_bundle.gzip_size(built[0]) <= 600_000
    monkeypatch.setattr(rl_bundle, "MAX_GZIP_BYTES", packed - 1)
    with pytest.raises(rl_bundle.BundleError, match="budget"):
        rl_bundle.check_size(built[0])


# --- 7. test-fold data only -----------------------------------------------------------------------


def test_curves_contain_test_fold_data_only(bundle: dict[str, Any]) -> None:
    rl_bundle.assert_curves_test_only(bundle)
    folds = bundle["meta"]["folds"]
    dates = bundle["curves"]["dates"]
    assert dates[0] == folds[0]["test"]["first_session"]
    assert dates[-1] == folds[-1]["test_last_mark_session"]
    assert dates[0] > folds[0]["purge"]["last_session"]
    replayed = [r["date"] for r in bundle["replay"]["records"]]
    assert replayed == dates[:-1]  # one decision per test bar; the last date is the final mark
    assert {b["kind"] for b in bundle["curves"]["background"]} == {"train", "val", "purge"}
    assert all(
        set(b) == {"fold_index", "kind", "first_session", "last_session"}
        for b in bundle["curves"]["background"]
    )  # spans only, never values


def test_a_train_region_leak_into_curves_fails(bundle: dict[str, Any]) -> None:
    leaked = json.loads(json.dumps(bundle))
    curves = leaked["curves"]
    train_day = leaked["meta"]["folds"][0]["train"]["last_session"]
    curves["dates"].insert(0, train_day)
    for series in curves["by_cost"].values():
        for values in series.values():
            values.insert(0, 1.0)
    for segment in curves["segments"]:
        segment["start"] += 1 if segment["start"] else 0
        segment["end"] += 1
    with pytest.raises(rl_bundle.BundleError):
        rl_bundle.assert_curves_test_only(leaked)

    widened = json.loads(json.dumps(bundle))
    fold = widened["meta"]["folds"][-1]
    fold["test"]["first_session"] = fold["purge"]["last_session"]  # test claims a purge day
    with pytest.raises(rl_bundle.BundleError):
        rl_bundle.assert_curves_test_only(widened)

    ragged = json.loads(json.dumps(bundle))
    next(iter(ragged["curves"]["by_cost"].values()))["agent"].append(1.0)
    with pytest.raises(rl_bundle.BundleError, match="values for"):
        rl_bundle.assert_curves_test_only(ragged)


# --- 8. adjustment provenance ---------------------------------------------------------------------


def test_adjustment_provenance_is_copied_verbatim(layout: Layout, bundle: dict[str, Any]) -> None:
    header = json.loads(layout.dataset.read_bytes())["header"]
    meta = bundle["meta"]
    assert meta["adjustment"] == header["adjustment"] == "all"
    assert meta["dataset_generated_at"] == header["generated_at"] == "2026-10-22T02:14:43.134336Z"
    assert meta["price_basis"] == header["price_basis"]
    assert meta["price_label"] == (
        "dividend-adjusted (total-return) closes as served by Alpaca on "
        "2026-10-22T02:14:43.134336Z; may reflect later corrections"
    )


# --- the exhibit, the eval table, the formats -----------------------------------------------------


def test_the_exhibit_is_one_labeled_seed_stitched_over_its_own_fold_policies(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    meta = bundle["meta"]
    assert meta["exhibit_seed"] == bundle["replay"]["exhibit_seed"] == 0
    assert "ONE training seed" in meta["exhibit_label"] and "5-seed" in meta["exhibit_label"]
    report = json.loads(layout.report.read_bytes())
    for item in bundle["policy"]["policies"]:
        key = f"walk_forward/fold{item['fold_index']:02d}/spy_ppo_seed0.policy.json"
        assert (
            item["artifact_sha256"]
            == report["artifact_sha256s"][key]
            == meta["policy_sha256s"][key]
        )
        fold = meta["folds"][item["fold_index"]]
        assert item["train"] == fold["train"] and item["test"] == fold["test"]
        records = [r for r in bundle["replay"]["records"] if r["fold_index"] == item["fold_index"]]
        assert records[0]["date"] == fold["test"]["first_session"]
        assert records[-1]["date"] == fold["test"]["last_session"]


def test_eval_table_carries_the_report_verbatim_and_the_honesty_block(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    report = json.loads(layout.report.read_bytes())
    table = bundle["eval_table"]
    vs_bh = next(
        g for g in report["gates"] if g["policy_id"] == "ppo" and g["name"] == "vs BUY-AND-HOLD"
    )
    assert table["agent"] == "ppo"
    assert table["headline_verdict"] == vs_bh["detail"].rsplit(" | ", 1)[-1]
    assert table["beats_buy_and_hold"]["ppo"] == table["headline_verdict"]
    assert table["headline_verdict"] == BEATS_BH_NO_CI_INCLUDES_ZERO  # this fixture's verdict
    assert table["n_folds"] == report["n_folds"] and table["n_seeds"] == report["n_seeds"] == 5
    assert table["break_even_cost_bps"] == report["break_even_cost_bps"]
    assert table["notes"] == report["notes"]
    assert table["caveats"]["look_count"] == 2
    assert (
        table["caveats"]["holdout"] is not None
        and "look-once holdout" in table["caveats"]["holdout"]
    )
    assert "AMENDED" in (table["caveats"]["gate_rule_amendment"] or "")
    assert [{k: v for k, v in g.items() if k != "verdict"} for g in table["gates"]] == report[
        "gates"
    ]
    words = {g["verdict"] for g in table["gates"]}
    assert words <= {"PASS", "FAIL", "VOID", "OK", "WARN", "YES", "NO"}
    for row, summary in zip(table["rows"], report["summaries"], strict=True):
        if row["sharpe_iqm_ci_source"] == "report":
            assert {k: v for k, v in row.items() if k != "sharpe_iqm_ci_source"} == summary


def test_the_one_allowed_recompute_reproduces_the_current_rl10_cis(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    rows = {(r["policy_id"], r["cost_bps"]): r for r in bundle["eval_table"]["rows"]}
    assert layout.computed_cis
    for key, ci in layout.computed_cis.items():
        row = rows[key]
        assert row["sharpe_iqm_ci_source"] == "recomputed"
        assert (row["sharpe_iqm_ci_low"], row["sharpe_iqm_ci_high"]) == ci
        assert row["sharpe_iqm_ci_note"] == "recomputed from per-fold rows (fold-only bootstrap)"
    assert {r["sharpe_iqm_ci_source"] for k, r in rows.items() if k not in layout.computed_cis} == {
        "report"
    }


def test_a_report_ci_that_its_own_rows_do_not_reproduce_is_refused(layout: Layout) -> None:
    report = json.loads(layout.report.read_bytes())
    row = next(s for s in report["summaries"] if s["policy_id"] == "ppo")
    row["sharpe_iqm_ci_low"] += 0.01
    with pytest.raises(rl_bundle.BundleError, match="not reproduced"):
        rl_bundle.eval_table(report, report_sha256="0" * 64, holdout_expected=True)


def test_f32_formats_are_lossless_and_refuse_non_float32() -> None:
    values = [_fround(v) for v in (0.1, -1.5e-7, 3.4e38, 0.0, -2.0, 1 / 3)]
    assert rl_bundle.decode_f32(rl_bundle.encode_f32(values), len(values)) == tuple(values)
    assert [_fround(rl_bundle.f32(v)) for v in values] == values
    assert len(repr(rl_bundle.f32(_fround(1 / 3)))) <= len("0.33333334")
    with pytest.raises(rl_bundle.BundleError, match="float32"):
        rl_bundle.encode_f32([0.1])  # a float64 that is not a float32
    with pytest.raises(rl_bundle.BundleError, match="bytes"):
        rl_bundle.decode_f32(rl_bundle.encode_f32(values), len(values) + 1)


def test_equity_path_mirrors_the_env_arithmetic() -> None:
    closes = [100.0, 101.0, 99.0, 102.0, 103.0]
    actions = [ENV_ACTION_ORDER[i] for i in (1, 1, 2, 0)]  # buy, (noop buy), sell, hold
    marks = rl_bundle.equity_path(closes, 0, actions, one_way_bps=5.0)
    e1 = 1.0 * (1.0 * (1.0 - 5.0 * 1e-4))  # flat over 0->1; buy fills at C1, pays cost
    e2 = e1 * (1.0 + (99.0 / 101.0 - 1.0))  # held over 1->2 (the second buy is a no-op)
    e3 = e2 * ((1.0 + (102.0 / 99.0 - 1.0)) * (1.0 - 5.0 * 1e-4))  # held 2->3, sell fills at C3
    assert marks == (1.0, e1, e2, e3, e3)


def test_stitch_and_band() -> None:
    assert rl_bundle.stitch([(1.0, 1.1), (1.0, 0.5, 2.0)]) == (1.0, 1.1, 1.1 * 0.5, 1.1 * 2.0)
    with pytest.raises(rl_bundle.BundleError):
        rl_bundle.stitch([(0.9, 1.0)])
    p25, p50, p75 = rl_bundle.quantile_band([(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.0, 4.0)])
    assert p50 == (1.0, 2.5) and p25 == (1.0, 1.75) and p75 == (1.0, 3.25)


def test_replay_starts_after_the_warmup_and_is_cost_free(bundle: dict[str, Any]) -> None:
    index = {bar["date"]: i for i, bar in enumerate(bundle["bars"])}
    assert min(index[r["date"]] for r in bundle["replay"]["records"]) >= WARMUP_BARS - 1
    assert bundle["features"]["window"] == WARMUP_BARS
    assert len(bundle["features"]["definitions"]) == bundle["policy"]["architecture"]["obs_dim"]


# --- review 2026-10-22: exhibit standing, the honest null band, the seed envelope ---------------


def _stitched_finals(report: dict[str, Any], policy_id: str, cost: float) -> dict[int, float]:
    """Per seed: the product of its per-fold final equities (what its stitched curve ends at)."""
    finals: dict[int, float] = {}
    for row in sorted(report["results"], key=lambda r: (r["seed"], r["fold_index"])):
        if row["policy_id"] == policy_id and row["cost_bps"] == cost:
            finals[row["seed"]] = finals.get(row["seed"], 1.0) * row["final_equity"]
    return finals


def test_exhibit_rank_is_computed_from_every_seeds_rows(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    report = json.loads(layout.report.read_bytes())
    rank = bundle["meta"]["exhibit_rank"]
    assert rank["exhibit_seed"] == 0 and rank["n_seeds"] == 5
    for cost_key, entry in rank["per_cost"].items():
        finals = _stitched_finals(report, "ppo", float(cost_key))
        assert {e["seed"]: e["stitched_final_equity"] for e in entry["seeds"]} == pytest.approx(
            finals, rel=1e-12
        )
        expected = 1 + sum(v > finals[0] for v in finals.values())
        assert entry["exhibit_rank_by_final_equity"] == expected
        for e in entry["seeds"]:
            sharpes = [
                r["sharpe"]
                for r in report["results"]
                if r["policy_id"] == "ppo"
                and r["seed"] == e["seed"]
                and r["cost_bps"] == float(cost_key)
            ]
            assert e["mean_fold_sharpe"] == pytest.approx(sum(sharpes) / len(sharpes), rel=1e-12)
    label = bundle["meta"]["exhibit_label"]
    assert "illustration, not the result" in label
    best = all(v["exhibit_rank_by_final_equity"] == 1 for v in rank["per_cost"].values())
    assert rank["best_by_final_equity_at_every_cost"] is best
    assert ("best of 5 seeds on this span" in label) is best


def test_the_label_says_best_of_n_when_the_exhibit_is_the_best_seed() -> None:
    best = {"n_seeds": 5, "best_by_final_equity_at_every_cost": True, "per_cost": {}}
    assert "best of 5 seeds on this span" in rl_bundle._standing(best)
    worse = {
        "n_seeds": 5,
        "best_by_final_equity_at_every_cost": False,
        "per_cost": {"0": {"exhibit_rank_by_final_equity": 3}},
    }
    assert "ranks 3 of 5 at 0 bps" in rl_bundle._standing(worse)


def test_the_headline_band_is_the_turnover_matched_gate_learn_null(
    layout: Layout, bundle: dict[str, Any]
) -> None:
    import statistics

    report = json.loads(layout.report.read_bytes())
    curves = bundle["curves"]
    assert curves["headline_band"] == "random_matched"
    assert curves["bands"]["random_matched"]["gate_learn_null"] is True
    assert "turnover-matched random (the GATE-LEARN null)" in curves["series"]["random_matched_p50"]
    assert "GATE-LEARN" in curves["semantics"]
    for cost_key, series in curves["by_cost"].items():
        cost = float(cost_key)
        for policy_id, prefix in (
            ("random_matched[ppo]", "random_matched"),
            ("random", "random_uniform"),
        ):
            finals = list(_stitched_finals(report, policy_id, cost).values())
            cuts = statistics.quantiles(finals, n=20, method="inclusive")
            for name, cut in zip(("p25", "p50", "p75"), (cuts[4], cuts[9], cuts[14]), strict=True):
                assert series[f"{prefix}_{name}"][-1] == pytest.approx(cut, rel=1e-5)
        agents = _stitched_finals(report, "ppo", cost)
        assert series["agent"][-1] == pytest.approx(agents[0], rel=1e-9)
        assert series["agent_seeds_min"][-1] == pytest.approx(min(agents.values()), rel=1e-5)
        assert series["agent_seeds_max"][-1] == pytest.approx(max(agents.values()), rel=1e-5)
        for low, high in zip(series["agent_seeds_min"], series["agent_seeds_max"], strict=True):
            assert low <= high


def test_the_uniform_band_is_labeled_and_dropped_first_when_the_budget_binds(
    layout: Layout, built: tuple[bytes, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    curves = built[1]["curves"]
    assert curves["bands"]["random_uniform"]["gate_learn_null"] is False
    assert "not the GATE-LEARN null" in curves["series"]["random_uniform_p50"]
    assert curves["dropped"] == []
    monkeypatch.setattr(rl_bundle, "MAX_GZIP_BYTES", rl_bundle.gzip_size(built[0]) - 1)
    data, smaller = job.build(job.parse_args(layout.argv(layout.root / "small.json")))
    assert rl_bundle.gzip_size(data) <= rl_bundle.MAX_GZIP_BYTES
    assert "random_uniform" not in smaller["curves"]["bands"] and smaller["curves"]["dropped"]
    for series in smaller["curves"]["by_cost"].values():
        assert "random_uniform_p50" not in series and "random_matched_p50" in series
        assert "agent_seeds_max" in series
    Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(json.loads(data))
    monkeypatch.setattr(rl_bundle, "MAX_GZIP_BYTES", 1000)
    with pytest.raises(rl_bundle.BundleError, match="budget"):
        job.build(job.parse_args(layout.argv(layout.root / "tiny.json")))


def test_f32_b64_wire_format_golden_vector() -> None:
    # The website decoder is written from the documented format; this pins it byte for byte.
    # 1.0 = 0x3F800000, -2.0 = 0xC0000000, float32(0.1) = 0x3DCCCCCD; little-endian per value,
    # then byte-plane grouped: [00 00 CD] [00 00 CC] [80 00 CC] [3F C0 3D].
    values = [1.0, -2.0, _fround(0.1)]
    golden = "AADNAADMgADMP8A9"
    assert rl_bundle.encode_f32(values) == golden
    assert rl_bundle.decode_f32(golden, 3) == tuple(values)
    import base64

    assert base64.b64decode(golden) == bytes.fromhex("0000cd0000cc8000cc3fc03d")


def test_honesty_caveats_fail_closed_after_more_than_one_look(layout: Layout) -> None:
    report = json.loads(layout.report.read_bytes())
    no_amendment = dict(report, notes=[n for n in report["notes"] if "AMENDED" not in n])
    with pytest.raises(rl_bundle.BundleError, match="amendment"):
        rl_bundle.eval_table(no_amendment, report_sha256="0" * 64, holdout_expected=True)
    no_holdout = dict(report, notes=[n for n in report["notes"] if "look-once holdout" not in n])
    with pytest.raises(rl_bundle.BundleError, match="holdout"):
        rl_bundle.eval_table(no_holdout, report_sha256="0" * 64, holdout_expected=True)
    first_look = dict(no_holdout, n_configurations_evaluated=1)
    table = rl_bundle.eval_table(first_look, report_sha256="0" * 64, holdout_expected=True)
    assert table["caveats"]["holdout"] is None


def test_a_stale_no_ci_note_is_replaced_when_the_cis_are_recomputed(layout: Layout) -> None:
    report = json.loads(layout.report.read_bytes())
    stale = (
        "IQM CIs are two-stage bootstraps (folds resampled, then seeds within each drawn fold); a "
        "policy with one run per fold has no CI. The gates read the fold-clustered CR0 SE with "
        "t(K-1)"
    )
    report["notes"] = [stale if n.startswith("IQM CIs") else n for n in report["notes"]]
    table = rl_bundle.eval_table(report, report_sha256="0" * 64, holdout_expected=True)
    assert stale not in table["notes"]
    assert rl_bundle.CI_NOTE_REPLACEMENT in table["notes"]
    assert len(table["notes"]) == len(report["notes"])
    assert any("EXCEPT its IQM-CI note" in n for n in table["bundle_notes"])


def test_curve_segment_labels_and_background_spans_must_match_the_folds(
    bundle: dict[str, Any],
) -> None:
    relabeled = json.loads(json.dumps(bundle))
    relabeled["curves"]["segments"][2]["first_session"] = relabeled["meta"]["folds"][2]["purge"][
        "last_session"
    ]
    with pytest.raises(rl_bundle.BundleError, match="labels"):
        rl_bundle.assert_curves_test_only(relabeled)
    shifted = json.loads(json.dumps(bundle))
    span = next(b for b in shifted["curves"]["background"] if b["kind"] == "val")
    fold = shifted["meta"]["folds"][span["fold_index"]]
    span["last_session"] = fold["purge"]["last_session"]  # val span stretched into the purge
    with pytest.raises(rl_bundle.BundleError, match="background"):
        rl_bundle.assert_curves_test_only(shifted)
