# RL Sleeve Program — `docs/RL-SLEEVE-PROGRAM.md`

**Status:** program spec, not yet implemented.
**Owner:** single-developer learning project + portfolio artifact.
**Branch target:** a new `research/rl-sleeve` branch off `ui/console-overhaul`.
**Last verified against live sources:** 2026-07-31.

---

## ⚠ 2026-10-21 RETARGET — the website is the deliverable (read before the July sections)

**The project's only requirement is now: a working RL demo displayable on stevenmurley.com.**
Everything below this addendum stands except where amended here. Decisions locked with the owner
on 2026-10-21:

1. **Demo format: interactive replay + live mini-trainer, fully static.** The exported policy
   (task-rl-09's torch-free MLP) runs IN THE VISITOR'S BROWSER (TypeScript forward pass) over the
   real bar history — scrubber, cost toggle (0/2/5 bps, recomputed client-side), action
   probabilities live. A second panel trains a tiny tabular Q-learner in-browser in seconds so
   visitors can watch learning happen. No backend; the page is static JSON + JS on Vercel.
2. **This repo goes PUBLIC.** The page links directly to the leakage suite, the env, and this
   document. Prerequisites before the flip (owner actions): git-history secret scan (see
   `docs/` scan report), push the branches, then flip visibility.
3. **Scope cut: task-rl-11 and milestones M8–M11 are DELETED.** No sleeve wiring, no cycle
   driver, no console attribution, no weekly cron. The platform sleeve demo served a meeting
   that no longer exists; the website serves everyone.
4. **Framing: lead with the honest null.** The page's headline is that the agent does NOT beat
   buy-and-hold and that proving this honestly is the point. §0 and §5 are unchanged and remain
   the heart of the project.

**Revised milestone tail** (M0–M7 unchanged):

| # | Milestone | Where | Demoable after? |
|---|---|---|---|
| **W0** | `task-rl-12` demo bundle job — one static `demo.json` (bars, exhibit replay, curves, eval table, policy weights + parity vectors) | this repo | bundle inspectable |
| **W1** | `/demo` page rebuild — replay scrubber, cost toggle, in-browser policy inference with load-time parity self-check, honest eval table, methodology strip with GitHub links. Replaces the Math.random fake. | `~/Desktop/personal-website` (Next.js 16 / Vercel) | ✅ **the deliverable** |
| **W2** | Live mini-trainer panel — in-browser tabular Q-learner on the same bars/discretized features/cost semantics, labeled a pedagogical toy distinct from the PPO exhibit | personal-website | ✅ wow factor |

**Two honesty rules the page must obey:**
- **Exhibit-seed rule:** the replay animates ONE designated seed's policy, labeled as such; every
  headline statistic remains the ≥5-seed, fold-clustered distribution from §5.
- **Mini-trainer rule:** the in-browser learner uses the same bar data and cost semantics as the
  real pipeline (discretized), and the page says plainly it is a teaching toy, not the PPO agent.

§8's terminal demo script is superseded by W1; it remains below as documentation of the offline
pipeline's entry points.

---

---

## 0. Honest expectations (read this first)

This sleeve is a **learning vehicle and an engineering demonstration**. The deliverable is a
correct, leak-free, honestly-evaluated reinforcement-learning loop wired into a real event-sourced
trading platform. The deliverable is **not** alpha.

### We expect the agent to NOT beat buy-and-hold out of sample. That is the expected result, not a failure.

Four independent reasons, each sourced:

**1. There is not enough data to tell.** Alpaca advertises **7+ years** of historical daily bars on
both the free and paid plans (the free/paid split is IEX vs SIP feed and the 15-minute SIP recency
rule, *not* lookback depth — <https://alpaca.markets/data>, <https://docs.alpaca.markets/us/docs/market-data-faq>).
That is roughly **1,760+ SPY daily bars**. Lo's standard error for a Sharpe estimate
(*The Statistics of Sharpe Ratios*, FAJ 58(4) 2002, eq. 9) is `SE(SR̂) ≈ sqrt((1 + SR²/2)/T)`, which
at 7 years of daily data gives **SE ≈ 0.38 on the annualized Sharpe**. Resolving a 0.3 Sharpe *gap*
versus buy-and-hold at 95% confidence needs roughly **43 years** in the most favorable case
(agent returns highly correlated with SPY) and **~85 years** if the two series were independent.
Any "we beat SPY by X%" number this project produces is statistically indistinguishable from luck
**by construction**. State that up front; do not discover it later.

**2. Seeds dominate the signal.** Henderson et al., *Deep Reinforcement Learning that Matters*
(arXiv:1709.06560) established that random seed alone produces non-overlapping learning curves in
deep RL. A recent trading-adjacent preprint (arXiv:2606.29194, §5.4 Table 3) reports a PPO baseline
whose Sharpe on the *same* holdout, *same* hyperparameters swings **+1.334 (seed 42) → −3.06
(seed 0)**, mean −0.755, std 2.206. Caveats we state honestly because they weaken the number: that
baseline is *AlphaGen pure PPO — Maskable PPO mining alpha expressions*, not an agent choosing
trading actions; it uses only 3 seeds; and the holdout is a 91-day segment whose Sharpe SE is
enormous on its own. Henderson is the load-bearing citation; 2606.29194 is a colorful illustration.
Either way: **we never report a single-seed number.**

**3. The action space caps the upside.** `TargetPosition.__post_init__` rejects negative quantity —
the platform is structurally **long-only**. On one ETF with a fixed position size the agent's entire
strategy space is "be in SPY or be in cash." Its only possible edge is **market timing**. Over most
historical SPY windows the optimal policy *inside that action space* literally is always-buy.
**If PPO converges to "always hold SPY," it learned the environment correctly.** We will not
reward-hack that away. Zhang, Zohren & Roberts (arXiv:1911.10107) say it plainly of their own
results: "most equity indices were dominated by large upward trends in the testing period," which is
why their long-only equity legs outperformed the RL methods.

**4. The closest published analogue is a null result.** Khare, Martheswaran & Dassanaike-Perera,
*Evaluation of Reinforcement Learning Techniques for Trading on a Diverse Portfolio*
(arXiv:2309.03202) run tabular RL (Value Iteration / SARSA / Q-Learning) on SPY. Result: with COVID
*in* the training set, "all models beat the baseline"; with COVID only in test, "holding the S&P 500
beat out all of our reinforcement learning strategies." Q-Learning was **best in-sample and worst
out-of-sample**. That is a regime result, not an alpha result.

And on what "winning" looks like even when the method is sound: Gort et al. (arXiv:2209.05559) run a
PBO hypothesis test (H=50 trials, N=5, k=2, J=10, α=10%). Plain walk-forward was rejected as overfit
(p=17.5%), SAC rejected (p=21.3%), **PPO accepted (p=8.0%)**, TD3 accepted (p=9.6%). In the 2022
crash window the *non-overfit* PPO agent returned **−34.96%**, versus SAC's −59.48% and the S&P BDM
index's **−50.78%**. The "winning" agent still lost a third of capital.

### Ship criteria — what "working" means here

| Tier | Criterion |
|---|---|
| **T0 — Mechanics** | `gymnasium.utils.env_checker.check_env` and `stable_baselines3.common.env_checker.check_env` both pass. PPO trains to completion. `ep_rew_mean` rises on the train split. The exported policy loads and returns a deterministic action for a given observation. |
| **T1 — Learned *a* policy** | Across ≥5 training seeds, the trained policy's action distribution is materially different from uniform-random, and its train-split reward beats random by more than the seed spread. It learned *something* — even if that something is "always buy." |
| **T2 — Doesn't fall apart OOS** | On held-out walk-forward folds (≥8 folds × ≥5 seeds), the agent's net-of-cost Sharpe is **not worse** than a turnover-matched random policy at 2 bps, reported as an interquartile mean with a fold-clustered CI. Never a point estimate. |
| **T3 — Explicitly OUT of scope** | "Consistently beats buy-and-hold after costs, out-of-sample, with a statistically distinguishable margin." We are not attempting this and will not claim it. |

**Paper only. `ALLOW_LIVE` stays off for this sleeve indefinitely.**

---

## 1. What this demonstrates to a reviewer

This is a portfolio piece. The headline is **rigor**, not returns. A reviewer at an AI lab reads
these signals, in roughly this order:

1. **Leakage discipline as a first-class, *tested* property.** Not a comment saying "no lookahead" —
   an executable proof. We ship a future-poisoning invariance test (mutate every bar after index *k*,
   assert every observation at steps ≤ *k* is bit-identical, including a NaN probe that makes any
   single arithmetic touch of a future bar fatal) **plus a negative control** (`_LeakyEnv` subclass
   that appends `close[t+1]` to the observation; the test suite asserts the detector *fails* on it).
   A leakage test that cannot fail is decoration. This is the single most differentiating artifact
   in the repo.
2. **PIT modeling of a real data quirk.** Alpaca stamps daily bars at the session's *opening*
   boundary (00:00 America/New_York → 04:00Z EDT / 05:00Z EST) while the value stored is that
   session's **close** (`infrastructure/market_data/alpaca_history.py:159` appends `entry["c"]`
   against `entry["t"]`; corroborated by the repo's own fixture
   `tests/unit/infrastructure/test_alpaca_historical_data.py:26`). Treating that timestamp as
   knowledge time reads ~16 hours into the future on *every step*. We carry an explicit `known_at`
   per bar and assert the invariant.
3. **Honest baselines through the same execution path.** Buy-and-hold, always-flat, random (30
   seeds, as a *null distribution* not a point), and SMA-cross — all expressed as `Strategy`
   implementations and scored through the identical replay. Almost no published paper ships a
   matched random-policy null band. It is the most informative plot we can produce.
4. **Walk-forward with purge and embargo, code-asserted.** Chronological folds, purge ≥ the longest
   feature lookback, non-overlapping test windows, a `fold.assert_no_leakage(obs_window=...)` that
   raises rather than a docstring that promises.
5. **Reward-hacking awareness, designed in.** Transaction costs are charged *inside the equity
   accounting*, so there is no `λ·turnover` weight anyone can quietly dial to zero. We report
   turnover, exposure fraction, action entropy, and `break_even_cost_bps` on every run, and we ship a
   degenerate-policy **veto** gate so "learned to sit still" can never be reported as a pass.
6. **Correct statistics for correlated observations.** We reuse `application/diagnostic.
   clustered_standard_error` but change the cluster key to the **walk-forward fold** — five seeds
   trained on the same price path are one draw of market luck, not five. We also state that the
   implementation is CR0 with **no G/(G−1) correction** and therefore use a `t(G−1)` critical value.
7. **Clean ports/adapters under a machine-checked contract.** PyTorch and stable-baselines3 never
   enter the application import graph — enforced by a new import-linter `forbidden` contract, not by
   discipline. The live inference path is ~40 lines of pure-stdlib Python (three matmuls, two tanh,
   one argmax) validated against SB3 by a 1,000-observation parity test.
8. **It runs in a real system.** Not a notebook. The policy trades in an event-sourced platform
   behind a non-bypassable risk gate, with orders tagged by `sleeve_id`, per-sleeve P&L attribution,
   and every decision reproducible from the event log plus a pinned policy hash.

### What a sharp reviewer will probe — and the answer

| Probe | Answer |
|---|---|
| "Show me the train/test split." | `application/rl_evaluation.make_folds` — rolling walk-forward, `train=504 / val=126 / purge=50 / test=63 / step=63`, non-overlapping test windows, `assert_no_leakage` raises. Never shuffled. |
| "How do you know there's no lookahead?" | `tests/unit/rl/test_env_leakage.py` — future-poison invariance (×3.0, ×0.25, NaN), reward-locality, a `price_at` ban assertion, and a `_LeakyEnv` negative control proving the detector has power. |
| "What's your baseline?" | Four, all run through the identical replay: buy-and-hold, always-flat, random ×30 seeds (a distribution), SMA(20,50) cross. Plus the frictionless `metrics.buy_and_hold_curve` number so execution drag is visible. |
| "How many seeds?" | ≥5 training seeds × ≥8 folds. Clustered by fold. The eval entry point *refuses to run* with fewer than 5 seeds. |
| "Did you tune on test?" | `n_configurations_evaluated` is incremented in the `user.action` audit event on every eval run; past 5 looks the report prints the Deflated-Sharpe caveat (Bailey & López de Prado, SSRN 2460551). The final 2 years are a look-once holdout. |
| "What transaction costs?" | `rl/costs.py::CostModel`, defaults sourced: SPY $0.01 spread on a ~$650 share ≈ 0.15 bps half-spread (SSGA, 2026-03-31); SEC §31 fee **$20.60 per $1M on sells, effective 2026-04-04** = 0.206 bps (<https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2>); FINRA TAF ≈ 0.000119/share. Default is ~1.0 bp one-way — deliberately 3–5× measured friction. Every headline metric is reported at 0 / 2 / 5 bps. |
| "Does your fill model flatter you?" | Yes, and we say so. `infrastructure/market_data/historical.py:31` returns `Quote(bid=price, ask=price, last=price)` — the backtest path is **zero-spread**, and `PaperBroker` stamps `fees=Decimal("0")`. So the env charges costs the replay does not; the report labels replay numbers as cost-free and publishes `break_even_cost_bps`. |
| "Why not just buy and hold?" | We report buy-and-hold on every row and expect it to win. The literal line `beats buy-and-hold: NO (CI includes 0)` prints unless the CI strictly excludes zero. |
| "Does torch ship to production?" | No. import-linter `forbidden` contract bans `trading.rl`, `torch`, `stable_baselines3`, `gymnasium` from every app module. Inference is pure stdlib, validated by a parity test. Render's build command installs `.[dev]`, never `.[rl]`. |
| "Is inference deterministic?" | Argmax over logits, never sampling — matches `PPO.predict(deterministic=True)` via `CategoricalDistribution.mode()`. The policy artifact's sha256 is recorded in the `DECISION_MADE` rationale, so any decision traces to the exact weights that made it. |

---

## 2. Locked decisions

These are settled. Design within them; do not relitigate.

1. **Objective:** get a running agent fast; learn by watching and tweaking. Profitability is a
   non-goal.
2. **Libraries:** `gymnasium` + `stable-baselines3` (PPO). No hand-rolled RL algorithms.
3. **First environment:** ONE ticker (SPY), **`Discrete(3)`** actions (flat / hold / long a fixed
   small position). Simplest correct loop.
4. **In-repo, isolated deps:** RL code under `backend/src/trading/rl/`; PyTorch/SB3 in an **optional
   dependency group** the app and the Render deploy never install; training runs as offline jobs
   like the existing backfill/label jobs; the live path does **inference only**.

Two decisions locked *by* this document as consequences:

5. **No `gym-anytrading`, no `FinRL`.** gym-anytrading's latest PyPI release is **2.0.0,
   2023-08-27**, last GitHub push 2024-03-14 — ~2.5 years stale, pinned to pre-1.0 gymnasium, pulls
   pandas + matplotlib, and has **no Hold action** (Sell=0/Buy=1), which its own README concedes "is
   a part of the main problem." FinRL's PyPI is frozen at **0.3.7, 2024-04-12** (GitHub is active,
   PyPI is not) and drags in a competing data + execution stack that would bypass this repo's ports,
   the PIT end-clamp, and the risk gate. Read gym-anytrading's `TradingEnv` as a reference; write
   ~250 lines in-repo.
6. **Torch-free live inference.** The deployable artifact is a JSON of the policy's weights, not the
   SB3 `.zip`. See §3.4.

---

## 3. Architecture

### 3.1 The core move: split RL into a HEAVY half and a PURE half

Do **not** lazy-import torch inside the strategy. Make the live path structurally incapable of
needing it.

```
backend/src/trading/
  domain/
    rl_features.py          [NEW] pure feature math, stdlib only.        ← ONE impl, shared
    rl_policy.py            [NEW] pure MLP forward + argmax, stdlib only. ← torch-free inference
  application/
    rl_dataset.py           [NEW] BuildRlDataset use case (HistoricalPricesPort -> RlDataset)
    rl_observation.py       [NEW] RlObservationFeed: async refresh, sync read
    rl_evaluation.py        [NEW] folds, gates, aggregation. stdlib only (mirrors diagnostic.py)
  infrastructure/
    rl/__init__.py          [NEW]
    rl/policy_store.py      [NEW] JSON artifact load/write + sha256
  strategies/
    rl_policy.py            [NEW] inference-only Strategy, @register("rl_ppo")
  rl/                       [NEW]  === HEAVY HALF: gymnasium / SB3 / torch live ONLY here ===
    __init__.py             docstring only; imports nothing
    bars.py                 PIT BarTable (known_at). stdlib only.
    costs.py                CostModel. stdlib only.
    rewards.py              log_return_reward + DifferentialSharpe. stdlib only.
    env.py                  SingleTickerTradingEnv(gym.Env)   [gymnasium + numpy]
    train.py                train_ppo(...)                     [stable_baselines3 + torch]
    export.py               sb3 -> domain MlpPolicy + parity   [stable_baselines3 + torch]
    baselines.py            buy_and_hold / always_flat / random / sma_cross
    scoring.py              rollout -> EquityPoints -> domain.metrics
  jobs/
    rl_dataset.py           [NEW] pull + freeze SPY bars to a local file
    rl_train.py             [NEW] offline PPO training
    rl_eval.py              [NEW] walk-forward evaluation report + gates
    rl_export.py            [NEW] SB3 .zip -> torch-free JSON artifact (+ parity assertion)
    rl_cycle.py             [NEW] the driver that actually makes the sleeve trade
  main.py                   [MOD] ~10 lines
  config.py                 [MOD] rl_* settings, all default-off
  interface/app.py          [MOD] 3 optional AppDeps fields
```

Splitting `bars.py` / `costs.py` / `rewards.py` / `rl_features.py` out of `env.py` is deliberate:
the leakage- and cost-critical arithmetic is then testable in the **default `pytest` run with zero
optional deps installed**.

### 3.2 Dependency isolation — the exact `pyproject.toml` deltas

```toml
[project.optional-dependencies]
dev = [
    # ... existing ...
    "numpy>=1.26,<3",
    "gymnasium>=1.3,<2",
]
rl = [
    "gymnasium>=1.3,<2",
    "stable-baselines3>=2.9,<3",
    "torch>=2.8,<3",
    "numpy>=1.26,<3",
    "tensorboard>=2.16",
]
```

**Why gymnasium goes in `dev`, not only in `rl`:** its wheel is **954 KB** and its deps are
`numpy>=1.21.0, cloudpickle>=1.2.0, typing-extensions>=4.3.0, farama-notifications` — no torch,
nothing heavy (<https://pypi.org/pypi/gymnasium/json>). In exchange, `gymnasium.Env` is a real,
`py.typed`, `Generic[ObsType, ActType]` class under `mypy --strict`, so
`class SingleTickerTradingEnv(gym.Env[Obs, np.int64])` type-checks properly and **never trips
`disallow_subclassing_any`** — which `strict` enables and which *would* fire if `gym.Env` resolved
to `Any`. Same reasoning for numpy (ships `py.typed`). Only the genuinely heavy pair stays `rl`-only:

```toml
[[tool.mypy.overrides]]
module = ["stable_baselines3.*", "torch.*"]
ignore_missing_imports = true

# strict enables warn_return_any; SB3/torch resolve to Any in CI where the rl extra is absent, so
# these two modules (the ONLY ones that touch SB3) get a narrow relaxation. Config override, NEVER
# an inline `# type: ignore` — strict also sets warn_unused_ignores, so an ignore that is REQUIRED
# without the extra becomes an ERROR once a developer installs it.
[[tool.mypy.overrides]]
module = ["trading.rl.train", "trading.rl.export"]
warn_return_any = false
disallow_untyped_calls = false
```

**Corollary:** do not subclass any SB3 type (e.g. `BaseCallback`) in `train.py`, or
`disallow_subclassing_any` returns. Use SB3's ready-made `CheckpointCallback` / `EvalCallback`
instances.

Make the isolation **enforced rather than accidental**. The existing `layers` contract lists only
`trading.{interface,infrastructure,application,strategies,domain}` — unlisted siblings
(`trading.jobs`, `trading.rl`, `trading.config`, `trading.main`) are unconstrained, so a new
`trading.rl` passes `lint-imports` by default and the isolation would rot. Add:

```toml
[tool.importlinter]
root_package = "trading"
include_external_packages = true      # REQUIRED to name torch/sb3/gymnasium in forbidden_modules

[[tool.importlinter.contracts]]
name = "Heavy ML deps never enter the app import graph"
type = "forbidden"
source_modules = [
    "trading.domain", "trading.application", "trading.strategies",
    "trading.interface", "trading.infrastructure", "trading.main",
]
forbidden_modules = ["trading.rl", "torch", "stable_baselines3", "gymnasium", "numpy"]
# trading.jobs is deliberately ABSENT from source_modules: jobs.rl_train MAY import trading.rl.
```

> ⚠️ `include_external_packages = true` changes how import-linter builds the graph for the **whole**
> config, including the existing layers contract. Run `lint-imports` immediately after adding it and
> **before** writing any RL code. Also: if any existing app module imports `numpy`, drop `numpy`
> from `forbidden_modules` (`application/diagnostic.py` documents "no numpy/pandas", so this should
> be clean — verify, don't assume).

Pytest:

```toml
markers = [
    # ... existing integration / live_smoke ...
    "rl: tests requiring the optional [rl] extra (torch/stable-baselines3); skipped without it",
]
```
guarded with `pytest.importorskip("stable_baselines3")`, so the default `pytest` run stays green on
a machine without torch.

`[tool.coverage.run] source = ["src/trading/domain", "src/trading/application"]` with
`fail_under = 80` means `src/trading/rl/**` is **outside** the denominator (good — the env won't drag
the gate) but `domain/rl_*.py` and `application/rl_*.py` **are inside** and need real tests.

**Render:** `docs/DEPLOY.md:9` describes install → migrate → `python -m trading.jobs.cycle`, and
there is no `render.yaml` in-repo — the build command lives in the Render dashboard.
**Verify by hand that it installs `.[dev]`-style and never `.[rl]`** before merging. Torch's Linux
wheels are the CUDA build (hundreds of MB to GBs); local training on Linux should use
`pip install --index-url https://download.pytorch.org/whl/cpu torch`, which cannot be expressed
portably in a PEP-621 extra and must be documented in the job's `USAGE::` docstring.

### 3.3 The strategy plug-in contract, and what it forces

From `strategies/base.py`:

```python
@runtime_checkable
class Strategy(Protocol):
    strategy_id: str
    def evaluate(self, signals: Sequence[Signal], portfolio: Portfolio) -> Sequence[StrategyProposal]:
        """Return absolute target positions. MUST be deterministic and side-effect free."""
```

`evaluate()` is **synchronous, pure, deterministic**, and receives only `(signals, portfolio)` — no
prices, no clock, no timestamp, no ports. And `trading.strategies` sits *below* `trading.application`
in the layers contract, so a strategy module may import **only** `trading.domain` (+
`trading.strategies`). It cannot import `HistoricalPricesPort`, `MarketDataPort`, or `trading.rl`.

Therefore the observation window and the policy must arrive at **construction time** as
domain-typed values / plain callables supplied by `main.py`. Modeled on `strategies/trivial.py`:

```python
# backend/src/trading/strategies/rl_policy.py
# Imports ONLY trading.domain.* + trading.strategies.*  — no ports, no torch, no numpy.
STRATEGY_ID: Final = "rl_ppo"
ClosesWindow = Callable[[], Sequence[float] | None]   # injected by main.py; None => inert

@dataclass
class RlPolicyStrategy:
    strategy_id: str = field(default=STRATEGY_ID, init=False)
    symbol: str = "SPY"
    position_size: Decimal = Decimal("1")
    policy: MlpPolicy | None = None            # trading.domain.rl_policy — pure, stdlib
    closes: ClosesWindow = lambda: None

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        # `signals` is IGNORED by design: this sleeve trades price, not disclosures.
        if self.policy is None:
            return ()
        window = self.closes()
        if window is None or len(window) < self.policy.features.window:
            return ()                                   # inert rather than guessing
        held = portfolio.quantity(self.symbol) > 0
        obs = build_observation(window, held=held, spec=self.policy.features)
        action = self.policy.act(obs)                   # deterministic argmax, never sampled
        if action is RlAction.BUY and not held:
            return (StrategyProposal(
                target=TargetPosition(self.symbol, self.position_size),
                rationale=f"rl_ppo {self.policy.policy_id} action=buy obs={_fmt(obs)}",
            ),)
        if action is RlAction.SELL and held:
            return (StrategyProposal(
                target=TargetPosition(self.symbol, Decimal("0")),
                rationale=f"rl_ppo {self.policy.policy_id} action=sell obs={_fmt(obs)}",
            ),)
        return ()                                       # HOLD, or a no-op buy/sell

@register(STRATEGY_ID)
def _build(config: StrategyConfig) -> RlPolicyStrategy:
    # registry.create() must be TOTAL: it cannot load an artifact, so it returns the inert form.
    # The real one is constructed in main.py (precedent: `political_strategy`).
    return RlPolicyStrategy()
```

Action → `TargetPosition` mapping, given `TargetPosition.quantity` is an **absolute target** and
`__post_init__` rejects `< 0`:

| Action | Proposal |
|---|---|
| `BUY` | `TargetPosition("SPY", position_size)` |
| `HOLD` | emit nothing (`()`), which yields delta 0 |
| `SELL` | `TargetPosition("SPY", Decimal("0"))` |
| *short* | **impossible** — negative quantity raises |

Downstream, `application/evaluate_signals.py` computes `delta = target.quantity - current`
(current includes pending open-order qty), derives the side, fetches a quote, appends
`DECISION_MADE` with `market_context`, runs the `RiskGate`, appends `RISK_EVALUATED`, and places.

### 3.4 Torch-free inference (the key isolation trick)

SB3's default `MlpPolicy` is exactly reproducible by hand. From
`stable_baselines3/common/policies.py`: `ActorCriticPolicy.__init__` defaults
`activation_fn = nn.Tanh` and `if net_arch is None: ... net_arch = dict(pi=[64, 64], vf=[64, 64])`.
From `torch_layers.py`: `MlpExtractor` builds `policy_net` as `Sequential(Linear, Tanh, Linear,
Tanh)`, so state_dict indices are `0` and `2`. For a 1-D `Box` observation the features extractor is
`FlattenExtractor` (`nn.Flatten()`, a no-op) and preprocessing is just `obs.float()`. For a
`Discrete` action space, `deterministic=True` → `CategoricalDistribution.mode()` → **argmax of the
logits**.

So the entire deterministic inference path is:

```
argmax( W3 @ tanh(W2 @ tanh(W1 @ obs + b1) + b2) + b3 )
```

~4,800 floats for a 9→64→64→3 net; ≈100 KB as JSON; ~40 lines of pure stdlib Python. State_dict keys
to export: `mlp_extractor.policy_net.0.{weight,bias}`, `mlp_extractor.policy_net.2.{weight,bias}`,
`action_net.{weight,bias}`.

```python
# backend/src/trading/domain/rl_policy.py — PURE. stdlib only. No numpy, no torch.
class RlAction(StrEnum):
    SELL = "sell"; HOLD = "hold"; BUY = "buy"

@dataclass(frozen=True, slots=True)
class DenseLayer:
    weight: tuple[tuple[float, ...], ...]      # [out][in], row-major (torch Linear.weight layout)
    bias: tuple[float, ...]
    activation: Literal["tanh", "relu", "identity"]

@dataclass(frozen=True, slots=True)
class MlpPolicy:
    policy_id: str
    schema_version: int
    features: FeatureSpec
    hidden: tuple[DenseLayer, ...]             # mlp_extractor.policy_net
    action_head: DenseLayer                    # action_net
    actions: tuple[RlAction, ...]              # index -> action
    def logits(self, observation: Sequence[float]) -> tuple[float, ...]: ...
    def act(self, observation: Sequence[float]) -> RlAction: ...   # argmax; NEVER samples
```

**The parity test is what makes this trustworthy.** `rl/export.py::assert_parity(model, policy,
observations)` samples ≥1,000 observations from the observation space and asserts
`PPO.predict(obs, deterministic=True) == policy.act(obs)` for every one. The training/export job
runs it over the full validation slice and **refuses to write the artifact on any disagreement**.

> Float-precision note: SB3 computes in float32 (`obs.float()`), the pure path in float64. Two logits
> within float32 epsilon could argmax differently. The parity test therefore asserts equality, and on
> any mismatch additionally asserts the top-2 logit margin is `< 1e-4` (a genuine tie) — and reports
> the count. Document the observed tie rate in the eval report.

`sb3_to_mlp_policy` must also assert `isinstance(model.policy.features_extractor, FlattenExtractor)`
and that every non-`Linear` module in `policy_net` is a supported activation — it refuses to export
anything the pure forward pass cannot reproduce.

### 3.5 Sleeve wiring — the one-line add point

`main.py:368-371` today:

```python
sleeve_specs: list[tuple[str, str, Strategy]] = [
    ("sleeve-political", "political_signal", political_strategy),
    ("sleeve-testing", "testing", strategy_registry.create("testing", strategy_config)),
]
```

Everything downstream (main.py:372-573) is a loop over `sleeve_specs`: `SleeveRegistry`
construction, per-sleeve `ModeRoutingBroker` + `EvaluateSignals` + `PrecheckRisk` + `SyncOrders`,
per-venue reconcilers keyed by sleeve mode, per-sleeve `ApplyTradeUpdate` fan-out. Adding the RL
sleeve is:

```python
# after historical_prices is built (main.py:~344-362)
rl_policy = load_policy(settings.rl_policy_path) if settings.rl_policy_path else None
rl_feed = RlObservationFeed(
    prices=historical_prices, clock=clock, symbol=settings.rl_symbol,
    window=rl_policy.features.window if rl_policy else _DEFAULT_RL_WINDOW,
)
rl_strategy = RlPolicyStrategy(
    symbol=settings.rl_symbol, position_size=settings.rl_position_shares,
    policy=rl_policy, closes=rl_feed.latest,
)
...
sleeve_specs = [
    ("sleeve-political", "political_signal", political_strategy),
    ("sleeve-testing", "testing", strategy_registry.create("testing", strategy_config)),
    ("sleeve-rl", "rl_ppo", rl_strategy),          # ← the one line
]
```

**Prefer a third sleeve over overwriting `strategies/testing.py`**, so the inert drop-in slot
survives for the next algorithm.

New `config.py` settings, all default-off so `python -m trading.main` and the existing test suite
are unchanged:

```python
rl_policy_path: Path | None = None            # None => the sleeve is INERT (no artifact needed)
rl_symbol: str = "SPY"
rl_position_shares: Decimal = Decimal("1")
rl_trade_mode: Literal["auto", "manual"] = "manual"
```

New `AppDeps` fields (following the `LabelOutcomes` / `run_backtest` precedent — a use case that
receives `prices=historical_prices`, rather than leaking the port itself):

```python
rl_observation: RlObservationFeed | None = None
rl_dataset: BuildRlDataset | None = None
rl_evaluator: Callable[[RlEvaluationRequest], Awaitable[RlEvaluationReport]] | None = None
```

### 3.6 Capital allocation — the one risk with real consequences

**`allocated_capital` is display-only.** `main.py:378` sets it to `settings.starting_cash` for every
sleeve, and `grep -rn allocated_capital src/` shows it used only in `interface/schemas.py:126,135,146`
and `application/backtest.py:107`. The risk gate sizes against `sleeve_equity = account.equity` —
the **whole shared Alpaca book** (`evaluate_signals.py:221`; `RiskContext.sleeve_equity` docstring:
"the shared pool equity for a live account"). There is exactly one global `RiskGate` built at
`main.py:234-246` and shared by all sleeves.

**Consequence: "give the RL sleeve a small allocation" is NOT achievable by config.** Anyone who
assumes otherwise hands an untrained PPO policy the full book. The RL sleeve's size is constrained
**only** by the small fixed `position_size: Decimal` inside the strategy (the `trivial.py` pattern).
`rl_position_shares = Decimal("1")` ≈ one SPY share ≈ $650 against a $100k paper book.

Corollary risk to accept consciously: the RL sleeve's drawdown consumes the *same* global
`max_drawdown_pct` breaker and the same `KillSwitch` as the political sleeve. Per-sleeve
`RiskLimits` is a separate, out-of-scope change.

### 3.7 The scheduled-execution gap

`python -m trading.jobs.cycle` runs **only** `deps.political_cycle`. `deps.evaluators` (keyed by
sleeve_id) is reached only from `POST /signals` (`interface/app.py:301`) and
`ExecuteProposals.execute(..., sleeve_id=...)`. **Without `jobs/rl_cycle.py` the RL sleeve appears
in `GET /sleeves` and never trades.** Do not treat sleeve wiring as "done."

`jobs/rl_cycle.py` sequence: `await deps.rl_observation.refresh()` → build the observation with the
same `build_observation` the env uses → `await deps.evaluators["sleeve-rl"].execute([], place=confirm)`.
Driving with an **empty signal list** is safe and correct: `ensure_real_signals([])` is vacuous and
`tradeable_signals([])` returns `[]` (both only filter the list they are given). Document that the
RL sleeve is driven with an empty signal list **by design**.

---

## 4. Environment spec

### 4.1 Pinned versions and the current API

| Package | Pin | Latest verified 2026-07-31 |
|---|---|---|
| gymnasium | `>=1.3,<2` | **1.3.0** (2026-04-22), `requires_python >=3.10` |
| stable-baselines3 | `>=2.9,<3` | **2.9.0** (2026-06-15), requires `gymnasium<2.0,>=0.29.1`, `torch<3.0,>=2.8`, `numpy<3.0,>=1.20`, `cloudpickle` |
| torch | `>=2.8,<3` | 2.13.0, cp312 wheels present |
| numpy | `>=1.26,<3` | 2.5.1 |

All compatible with the repo's `requires-python = ">=3.12"`.

> The `gymnasium<2.0` ceiling is SB3's, and was only widened from `<1.3.0` on 2026-06-15. SB3 tracks
> gymnasium minors with a one-release lag. **Re-check the pair before bumping either.**

**Gymnasium `Env` API** (frozen since 1.0.0 — <https://gymnasium.farama.org/api/env/>):

```python
reset(*, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]
step(action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]  # obs, reward, terminated, truncated, info
```
`super().reset(seed=seed)` is **mandatory** — it seeds `self.np_random`. `reward_range` was
deprecated in 0.29 and removed from the 1.x Env API: **do not set it**.

**SB3 `PPO`** (<https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>):

```python
PPO(policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
    gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None,
    rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None,
    policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)
learn(total_timesteps, callback=None, log_interval=1, tb_log_name='PPO',
      reset_num_timesteps=True, progress_bar=False)
predict(observation, state=None, episode_start=None, deterministic=False) -> (action, state)
save(path, exclude=None, include=None)
PPO.load(path, env=None, device='auto', custom_objects=None, print_system_info=False, force_reset=True)
```
Discrete action spaces are supported; SB3 requires `Discrete(n)` with `start=0`. Docs note PPO "is
meant to be run primarily on the CPU" — no GPU needed. Validation:
`stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)`.

### 4.2 PIT bar table — the load-bearing assumption

```python
# backend/src/trading/rl/bars.py — stdlib ONLY (no numpy, no gymnasium)
_SESSION_LENGTH = timedelta(hours=16)      # 00:00 ET -> 16:00 ET close
_SETTLE_BUFFER  = timedelta(minutes=15)    # mirrors alpaca_history._RECENT_DATA_BUFFER (16 min)

@dataclass(frozen=True, slots=True)
class Bar:
    session_ts: datetime   # tz-aware UTC; Alpaca's `t` (session OPEN boundary)
    close: Decimal
    known_at: datetime     # tz-aware UTC; when `close` became public
    # __post_init__ rejects naive datetimes (ruff DTZ) and known_at <= session_ts

@dataclass(frozen=True, slots=True)
class BarTable:
    symbol: str
    bars: tuple[Bar, ...]                  # strictly ascending by session_ts
    @classmethod
    def from_price_series(cls, series: PriceSeries, symbol: str, *,
                          session_length: timedelta = _SESSION_LENGTH,
                          settle_buffer: timedelta = _SETTLE_BUFFER) -> BarTable: ...
    def closes(self) -> tuple[float, ...]: ...
    def slice_known_by(self, when: datetime) -> BarTable: ...
    def split(self, at: datetime, *, embargo: timedelta) -> tuple[BarTable, BarTable]: ...
```

`known_at = session_ts + 16h + 15min`. Because `session_ts` is local-midnight already expressed in
UTC (05:00Z in EST / 04:00Z in EDT), a **fixed** 16h15m offset is correct in both DST regimes.

> **This is an assumption verified by inference, not by Alpaca's docs.** The reference page documents
> `t` only as "Timestamp in RFC-3339 format with nanosecond precision" and never states begin-vs-end
> (<https://docs.alpaca.markets/us/reference/stockbars>). The repo's own fixture corroborates it
> (`{"t": "2026-01-01T05:00:00Z", ..., "c": 100.5}`). **Do a one-off live check against a real SPY
> bar before any `--confirm` run.** If Alpaca ever switches to end-of-bar stamping, the fixed offset
> becomes a 16-hour *backward* error and every internal-consistency test still passes.
>
> Early-close sessions (day after Thanksgiving, Christmas Eve, 13:00 ET) make `known_at` ~3 hours
> *late* — conservative, therefore safe. Keep the offset conservative and never shorten it.

**Two hard bans, asserted in tests:**
- The env must **never** call `PriceSeries.price_at`. Verified in source
  (`domain/prices.py:16-25`): `index = bisect_right(times, when) - 1; if index < 0: return
  points[0][1]` — it returns the **earliest** point when `when` precedes all points, a silent
  left-edge forward-look with no error and no warning. Slice by integer index.
- The env must **never** treat `session_ts` as knowledge time.

**Bar adjustment:** the adapter sends `{symbols, timeframe, start, end}` and never `adjustment`;
Alpaca's default is **`raw`** (unadjusted), options `raw|split|dividend|spin-off|all`. SPY has had no
split since 2005, so within a 2016+ window this is a dividend-only distortion (~1.3%/yr) that hits
the agent and buy-and-hold **identically** — internally consistent, but total returns are understated
for both. Either add an `adjustment` kwarg to `AlpacaHistoricalData` and pass `all` for the RL pull,
or accept `raw` and state it in the report. **Do not leave it unstated.**

### 4.3 Observation — `Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32)`

| idx | feature | scaling |
|---|---|---|
| 0 | `log(C_t / C_{t−1})` | `/ 0.01` |
| 1 | `log(C_t / C_{t−5})` | `/ (0.01·√5)` |
| 2 | `log(C_t / C_{t−21})` | `/ (0.01·√21)` |
| 3 | `C_t / SMA_10 − 1` | `/ 0.02` |
| 4 | `C_t / SMA_50 − 1` | `/ 0.05` |
| 5 | `realized_vol_21 / 0.01 − 1` | — |
| 6 | `SMA_10 / SMA_50 − 1` | `/ 0.03` |
| 7 | **position flag** ∈ {0.0, 1.0} | — |
| 8 | **unrealized P&L** `(C_t/entry − 1) / 0.05`, 0.0 when flat | — |

All clipped to ±10 so `check_env`'s dtype-exact containment assertion holds
(`.astype(np.float32, copy=False)` after the clip).

Four non-negotiables:

- **Indices 7 and 8 are not optional.** Without the position in the observation the process is not
  Markov — the agent cannot know whether BUY is legal or what it is sitting on, and the value
  function becomes unlearnable noise. This is the most common silent bug in hobby trading envs.
- **Normalization uses FIXED a-priori constants, not fitted statistics.** A `StandardScaler` or
  z-score fitted over the dataset leaks the future's mean and variance into every observation — an
  invisible lookahead that no index-checking test catches. The constants above are chosen from ~1%/day
  equity vol and live in a frozen `FeatureScales` dataclass embedded in the artifact. (Documented
  alternative if you later want adaptive scaling: fit on the **train slice only** and freeze into
  `FeatureSpec` — never a trailing full-sample statistic.)
- **Do NOT use `VecNormalize`.** Its running obs/reward stats keep updating (a leak unless you set
  `training=False, norm_reward=False`), must be saved/loaded as a second artifact, and would have to
  be replicated byte-identically in the torch-free live path — defeating §3.4. Normalizing inside the
  env with stateless causal math means training and production compute the *same function*.
- **No raw price level, no calendar/date feature.** A raw level is non-stationary and lets the net
  memorize "which year is this"; a date index lets it memorize the train/eval split.

`build_observation` takes a **slice** (`closes[: t+1]`), never `(full_list, index)`. The future is
then physically absent from the function's inputs, so leakage is impossible by construction rather
than by review. `WARMUP_BARS = 50` (SMA_50 is the longest lookback); first legal decision index is 49.

**One implementation, shared by the env and the live strategy** (`domain/rl_features.py`). This is
the train/serve-skew firewall.

### 4.4 Action — `Discrete(3)`

`0 = HOLD, 1 = BUY, 2 = SELL`, with HOLD at index 0 so an untrained/degenerate policy defaults to
inert. Long-only, one fixed position size (all-in / flat on the env's scale-free account).

Illegal actions (BUY while held, SELL while flat) **resolve to HOLD**, `info["action_was_noop"] =
True`, and `invalid_action_penalty` defaults to **0.0**. Penalizing invalid actions is a known route
to degenerate policies; the no-op is already implicitly penalized by the absence of profit. Keeping
it a config knob makes it a *learning lever* — flip it to 0.001 and watch the action distribution
shift. The principled upgrade is action masking via `sb3-contrib`'s `MaskablePPO`; note it in the
docstring, don't add the dep.

Keeping an explicit **HOLD** is deliberate: gym-anytrading's omission of it is a documented weakness
its own README concedes.

### 4.5 Reward — log return of net-of-cost equity, ×100

```
growth  = 1 + position_t · (C_{t+1}/C_t − 1) − cost_paid_fraction
reward  = 100 · ln(max(growth, 1e-9))
```

Costs are charged **inside the equity accounting**, not as a separate `−λ·turnover` term. That is the
whole point: **there is no shaping weight anyone can dial to zero.** The only remaining knob is
`CostModel`, which is a *physical claim about the market* and is therefore sweepable, not tunable.

Why this reward:
- **Auditable.** `Σ reward / 100 = ln(E_T / E_0)`. Compare the episode return to buy-and-hold on the
  same window and a broken env is immediately visible.
- **Hard to hack.** The only ways to raise it are be long when the market rises and flat when it
  falls. Churn strictly loses (costs). Size doesn't help (equity-normalized). Inactivity earns 0.
- **Well-scaled.** ×100 makes daily rewards ~±2, an affine rescale that leaves the optimal policy
  identical — so we keep interpretability instead of reaching for `VecNormalize(norm_reward=True)`.
- **Concave in wealth** — free, hyperparameter-less risk aversion; a −50%/+100% round trip scores 0.

Moody & Saffell's original additive-profit objective (NeurIPS 1998, eq. 1) includes the `δ|F_t −
F_{t−1}|` cost term for exactly this reason; Zhang et al. (arXiv:1911.10107, eq. 4) state it
directly: "*We can treat bp as a regularising term as a large bp penalises turnovers and lets agents
maintain current trade positions.*"

**Experiment #2, behind a flag:** `--reward {log_return,dsr}` implementing Moody & Saffell's
differential Sharpe (eqs. 4–5):
`D_t = (B_{t−1}·ΔA_t − ½·A_{t−1}·ΔB_t) / (B_{t−1} − A²_{t−1})^{3/2}`, η ≈ 0.02–0.04, with two
**mandatory guards**: append `(A_{t−1}, B_{t−1})` to the observation (the reward depends on hidden
EMA state, so the MDP is otherwise **not Markov**), and return 0.0 when `B − A² ≤ 1e-12` (the
denominator → 0 for a do-nothing policy, making "never trade" an exploding attractor). Comparing the
two rewards on identical seeds and folds is the single most instructive tweak in the project.

### 4.6 Cost model

```python
# backend/src/trading/rl/costs.py — PURE stdlib. Importable by CI without the rl extra.
@dataclass(frozen=True, slots=True)
class CostModel:
    half_spread_bps: float = 0.10          # $0.01 spread on a ~$650 SPY share ≈ 0.15 bps
    slippage_bps: float = 0.65             # HONESTY BUFFER — see below
    sec_fee_bps_on_sells: float = 0.206    # SEC §31: $20.60/$1M, effective 2026-04-04. CHANGES.
    taf_per_share_on_sells: float = 0.000119   # FINRA TAF
    commission_per_trade: float = 0.0      # Alpaca US equities are commission-free
    def effective_price(self, *, side: Side, price: float) -> float: ...
    def fees(self, *, side: Side, shares: float, price: float) -> float: ...

ZERO_COST   = CostModel(0.0, 0.0, 0.0, 0.0, 0.0)   # ABLATION ONLY
STRESS_COST = CostModel(slippage_bps=4.9)          # ~5 bps one-way robustness sweep
```

Measured SPY friction is ~0.4 bps round trip. The default (~1.0 bp one-way ≈ 2 bps round trip) is
deliberately 3–5× that, because **neither simulator charges anything realistic**:
- **Alpaca paper trading** fills at NBBO and explicitly does not simulate market impact, information
  leakage, latency slippage, queue position, price improvement, regulatory fees, or dividends; order
  size is not checked against NBBO size; partial fills occur randomly ~10% of the time
  (<https://docs.alpaca.markets/us/docs/paper-trading>).
- **In-repo `PaperBroker`** buys at ask / sells at bid (`paper.py:58`) but stamps `fees=Decimal("0")`
  (`paper.py:88`), full instant fills.
- **The replay path is fully frictionless**: `infrastructure/market_data/historical.py:31` returns
  `Quote(symbol=symbol, bid=price, ask=price, last=price, ...)` — bid == ask == close.

**Every rate is a dated `CostModel` default with a source comment, never an inline literal** — the
SEC §31 rate was $0.00/million through 2026-04-03 and $20.60/million from 2026-04-04.

Cost arithmetic that justifies the default: 2 bps round-trip on a fully-invested notional costs
≈ 5.0%/yr if the agent flips daily — comparable to half the equity premium, so churn is visibly
punished but not instantly fatal. At 10 bps the drag is ≈25%/yr and only buy-and-hold survives. The
first is the training default; the second is the stress test.

> **Env sizing is coupled to cost realism.** The env trades **all-in / flat on a scale-free simulated
> account** (BUY = 100% of env cash into SPY). If the env instead traded a small fixed share count
> (e.g. 1 SPY ≈ 0.65% of $100k), the 2 bps charge would cost ~0.03%/yr in equity terms and the cost
> term would stop teaching anything. Real sizing happens in the **live sleeve** (`position_size`), not
> in the reward. The policy is scale-free (no notional in the observation) so this train/live scale
> mismatch is safe — but it must be stated in the env docstring, and **live realized P&L will not
> equal env reward.**

### 4.7 Episode structure and the timing contract

```
Bar universe : SPY daily, ~7-8 years (≈1,760-2,000 bars) from jobs/rl_dataset.py, frozen to disk
Episode      : 252 contiguous decision steps (≈1 trading year)
Training     : random_start=True — self.np_random.integers(lo, hi+1). NEVER `random`.
Evaluation   : random_start=False, explicit options={"start_index": ...}, NON-OVERLAPPING windows
terminated   : equity <= ruin_fraction (0.2)  — the only genuine absorbing state
truncated    : steps >= episode_length or cursor >= len(bars) - 1
```

**Getting `terminated`/`truncated` backwards is a real bug, not pedantry.** PPO bootstraps `V(s')`
on truncation but not on termination. Reporting step-budget exhaustion as `terminated` tells the value
function the world ends every 252 days and systematically biases it downward.

**Random episode starts are mandatory.** A single-ticker daily series is one deterministic
trajectory; without random start offsets every rollout is identical and PPO memorizes rather than
learns. This is the most common way this project produces a nonsense agent. Build it into `reset()`
from day one.

**The timing contract (stated in the module docstring, asserted in tests):**

> At step *t* the agent stands just after the close of session *t*. Its observation is built ONLY
> from `closes[: t+1]`, every one of which has `known_at <= bars[t].known_at`. The action is filled
> at the close of session *t+1*, and the reward is the portfolio log return over *t → t+1*.
> `bars[t+1]` is read **exactly once**, in `step`, to score the action; it never reaches `_observe`.
> Removing that one-bar lag turns the agent into an oracle rewarded for a return it already observed.

`EnvConfig` exposes `execution_lag: int = 0` as an additional knob; report results at lag 0 **and**
lag 1. If the edge vanishes at lag 1, there was no edge.

### 4.8 Leakage guards — the test suite that makes this credible

`backend/tests/unit/rl/test_env_leakage.py` (needs gymnasium+numpy, which are in `dev` — runs in the
default `pytest`):

**Test A — future-mutation invariance (the decisive one).** Don't inspect indices; make leakage
impossible to hide.

```python
def _poison(closes: list[float], k: int, *, factor: float) -> list[float]:
    """Replace every close AFTER index k. Anything the env legitimately sees is untouched."""
    return [c if i <= k else c * factor for i, c in enumerate(closes)]

@pytest.mark.parametrize("k", [60, 120, 300])
@pytest.mark.parametrize("factor", [3.0, 0.25, float("nan")])
def test_observation_is_invariant_to_every_future_bar(k: int, factor: float) -> None:
    ...  # step clean and poisoned envs in lockstep; assert np.array_equal before every step;
         # break once cursor > k; assert not np.isnan(obs).any()
```
`factor=nan` is the strongest probe: a single arithmetic touch of a future bar makes the observation
NaN and both assertions fire. `3.0` / `0.25` catch touches that survive as finite numbers (a
dataset-wide scaler, a centred rolling mean, a `.mean()` over the full array). One test covers a
whole *class* of bug.

**Test A′ — the negative control. Do not skip this.** A leakage test that cannot fail is decoration:

```python
class _LeakyEnv(SingleTickerTradingEnv):
    def _observe(self):
        obs = super()._observe()
        obs[0] = np.float32(self._closes[self._cursor + 1] / self._closes[self._cursor])
        return obs

def test_leakage_detector_actually_detects_leakage() -> None:
    with pytest.raises(AssertionError):
        _run_invariance_check(_LeakyEnv, k=120, factor=3.0)
```

**Test B — `known_at` ordering.** With `audit=True`, every bar read while building an observation
must satisfy `known_at <= env.decision_at` (captured *before* stepping).

**Test C — reward alignment / no free lunch.** On a deterministic series, always-BUY reproduces
buy-and-hold to within 1e-9 at `cost_bps=0` (`exp(Σ reward/100) == close[T]/close[0]`). And: two
series identical except `closes[t+1]` produce the **same observation** at *t* and **different
rewards** — proving the lag exists in the direction we want (reward forward, observation backward).

**Test D — split hygiene** (`test_bars_pit.py`, stdlib only): embargo prevents window straddle;
train/eval session sets are disjoint; `known_at` is after the session close under **both** DST regimes
(`2026-01-02T05:00:00Z → 21:15:00Z`; `2026-07-02T04:00:00Z → 20:15:00Z`).

**Test E — architectural.** `assert "price_at" not in Path("src/trading/rl/env.py").read_text()`,
with a comment citing `domain/prices.py:23-24`. Cheap, and it stops a future contributor from
"simplifying" the env back into the bug.

**Test F — SB3 contract** (`@pytest.mark.rl`): both `gymnasium.utils.env_checker.check_env` and
`stable_baselines3.common.env_checker.check_env`, plus a determinism test (two `reset(seed=42)` give
identical `start_index` and identical observations).

### 4.9 PPO hyperparameters for the first run

```python
PPO("MlpPolicy", Monitor(env, info_keywords=("equity", "turnover", "n_trades")),
    n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, learning_rate=3e-4, seed=<explicit>, device="cpu", verbose=1,
    tensorboard_log=...)
model.learn(total_timesteps=200_000)
```

Deviate from SB3 defaults in exactly three places: `n_steps=1024` (the default 2048 is ~8 episodes
per rollout at `episode_length=252`; 1024 is ~4, giving more frequent updates on a tiny dataset),
`ent_coef=0.01` (default 0.0; a little entropy stops the discrete policy collapsing to always-HOLD in
the first 10k steps), and an **explicit seed**. `device="cpu"` — a 64×64 MLP is faster on CPU than
GPU, and SB3's docs say PPO "is meant to be run primarily on the CPU."

`Monitor` **before any other wrapper** (SB3 docs), with `info_keywords` so financial metrics come out
of the env for free. `γ=0.99` is a deliberate bias away from the terminal-log-wealth objective the
additive reward defines — itself a good "tweak and watch" knob (Zhang et al. used γ=0.3).

Sanity note: daily log rewards ×100 are ~±2, episode returns ~±10-50. If the learning curve looks
flat, suspect reward scale before the policy; expose `reward_scale: float = 1.0` as a **diagnostic**,
not a default.

---

## 5. Evaluation

### 5.1 Two tiers

**Tier A — fast, inside training only.** `EvalCallback(eval_env, n_eval_episodes=5, eval_freq=...,
deterministic=True, best_model_save_path=...)` on the fold's **validation** slice. Used for early
stopping and model selection ONLY. Never quoted as a result. Never touches the test slice.

**Tier B — authoritative.** The exported torch-free policy replayed over the fold's **test** slice
through `rl/scoring.py`, feeding the existing pure functions in `domain/metrics.py`. Every baseline
runs through the identical call. This is the only tier the gates read.

**Optional Tier B′ (high demo value, low cost):** replay the policy through
`application/backtest.Backtest`, which exercises the *real* `EvaluateSignals → RiskGate → PlaceOrder`
path via `InMemoryBacktestEnvFactory`. Divergence between the env's cumulative reward and the
backtest equity curve is itself a bug detector.

> **Blocker if you do this:** `Backtest.run` is **signal-driven** — `for when, group in
> groupby(sorted(request.signals, key=created_at), ...)` — it advances the clock only to timestamps
> that carry a signal. A signal-agnostic RL strategy gets **zero decision points** and silently
> reports a flat curve that looks like a valid "the agent did nothing" result. Either synthesize one
> heartbeat `Signal` per decision bar (SYNTHETIC provenance is fine — `Backtest` builds its evaluator
> **without** `require_real_data`; just never BACKFILLED, which `tradeable_signals` always drops) or
> write a ~60-line RL replay loop over `InMemoryBacktestEnvFactory`. Either way, **assert
> `result.decision_count == expected_bars`.**

### 5.2 Baselines — four, all as `Strategy`, sharing the agent's action set and `position_size`

`backend/src/trading/strategies/rl_baselines.py` (imports only `trading.domain` +
`trading.strategies`; zero torch):

| id | policy | why |
|---|---|---|
| `rl_bh` | first decision → long; hold to the end | the real bar |
| `rl_flat` | never trades | anchors the cost/turnover floor and detects "learned to sit still" |
| `rl_random` | uniform over {BUY, HOLD, SELL}, seeded | the **null distribution** — run **30 seeds per fold** |
| `rl_ma` | long when SMA_20 > SMA_50, else flat | the classic market-timing rule the agent actually competes with |

Also report the frictionless `metrics.buy_and_hold_curve` number alongside the `Strategy` version so
execution drag is visible, and (if cheap) an **untrained policy at initialization**
(`total_timesteps=0`) to separate "PPO learned" from "this env is trivially profitable."

**The random-policy band is the single most informative plot.** If the agent's OOS return sits inside
the monkeys' 5th–95th percentile, you have learned nothing and the report says so.

### 5.3 Walk-forward split

Rolling (not anchored — fixed train size keeps folds exchangeable, which the clustered SE assumes):

```
train  = 504 bars (2y)
val    = 126 bars (6m)     # EvalCallback / early stopping only
purge  = 50 bars           # == WARMUP_BARS: a 50-bar feature window must not straddle the split
test   =  63 bars (1 quarter)
step   =  63 bars          # test windows NON-OVERLAPPING -> folds are independent price paths
```

Fold *k*: `train=[i, i+504)`, `val=[i+504, i+630)`, gap `[i+630, i+680)`, `test=[i+680, i+743)`,
`i += 63`. Over ~1,760 bars this yields **K ≈ 17 folds**; over ~2,500 bars, K ≈ 28. Assert `K >= 8`
or the report is marked `low_n` and **no gate may pass** (mirroring the repo's
`_MIN_EPISODE_CELL_N = 30` `low_n` convention).

> `step_bars == test_bars` is not optional. If `step < test`, folds share bars and the fold-clustered
> SE is too tight.

Retrain PPO once per fold (200k timesteps on 504 bars is seconds-to-minutes on CPU; ~17 folds × 5
seeds = 85 trainings is a coffee break, not a cluster job).

Leakage rules **asserted in code, not comments**: env observations built by integer index slicing;
`price_at` forbidden inside `trading.rl`; feature scaling by fixed constants; and per fold,
`fold.assert_no_leakage(obs_window=WARMUP_BARS)` raising `LeakageError`.

### 5.4 Metrics

Straight off `domain/metrics.py` (pure stdlib, already present — do **not** rebuild):
`total_return_pct`, `cagr_pct`, `sharpe_ratio` (rf=0, population stdev, `DEFAULT_ANNUALIZATION =
252.0`), `sortino_ratio`, `return_volatility`, `max_drawdown_pct`, `period_returns`, `round_trips`
(FIFO) → `trade_stats` (win_rate, profit_factor), `buy_and_hold_curve`, `benchmark_total_return_pct`.

Three new pure functions in `domain/metrics.py` (inside the 80% coverage denominator → they need
unit tests):

```python
def annualized_turnover(trades, curve, *, annualization: float = DEFAULT_ANNUALIZATION) -> float
def exposure_fraction(curve, positions) -> float          # 1.0 == buy-and-hold in disguise
def cost_adjusted_returns(returns, trades, curve, *, cost_bps: float) -> tuple[float, ...]
```

Report every headline metric at **`cost_bps ∈ {0, 2, 5}`** plus `break_even_cost_bps` — the cost
level at which the agent's edge over random crosses zero. A result that only survives at 0 bps is not
a result. **Always** report action-distribution stats: fraction of steps long, position flips,
longest holding streak, action entropy. That is how you detect "collapsed to always-buy" /
"collapsed to always-flat" / "churning noise" at a glance.

Also report `risk_clipped_fraction` — if the `RiskGate` clipped more than ~10% of orders you are
evaluating the risk gate, not the agent, and `position_size` must come down.

### 5.5 Aggregation — clustering, done correctly

`application/diagnostic.clustered_standard_error(values_by_cluster: Mapping[str, Sequence[float]]) ->
float | None` is the directly reusable primitive. `build_report` / `form4_episode_stats` are
hard-wired to Form-4 signals + `OutcomeLabel` and are **not** reusable.

**Cluster key = the walk-forward fold**, one observation per `(fold, seed)`. Five seeds trained on
fold *k* all trade the *same* price path — that is ~one independent draw of market luck, not five.
CR0-by-fold is exactly the estimator that gives you ≈`sqrt(K)`, not `sqrt(K·S)`. Clustering by ticker
degenerates to **G=1** on a single-ticker agent and produces a confident-looking, meaningless SE.

> ⚠️ The implementation is **CR0 with no G/(G−1) correction** (verified in the docstring at
> `diagnostic.py:426`). With G ≈ 8–17 folds that is **anti-conservative**. Use a `t(G−1)` critical
> value instead of 1.96, and say so in the report.

Report **interquartile mean + stratified-bootstrap 95% CI across (fold, seed)** (Agarwal et al.,
NeurIPS 2021, arXiv:2108.13264 — "reliable evaluation in the few run deep RL regime cannot ignore the
uncertainty in results") **alongside** mean ± clustered SE, plus a per-fold scatter of agent vs each
baseline. Never report the best seed or the best fold.

### 5.6 The "did it learn?" gates

Let `S_agent[k,s]` = net (2 bps) Sharpe of the agent on fold *k*, seed *s*; `S_rand[k]` = mean net
Sharpe of the 30 random seeds on fold *k*; `d[k,s] = S_agent[k,s] − S_rand[k]`.

**GATE-NOTDEGENERATE — a veto, evaluated FIRST.** If it fails, GATE-LEARN reports **VOID**, never
PASS:
- `0.05 <= exposure_fraction <= 0.95`, AND
- action entropy over test bars `> 0.2 nats`, AND
- `|corr(agent position path, buy-and-hold position path)| < 0.98`.

Rationale: `sharpe_ratio()` returns **`0.0`, not NaN**, on a flat curve, so an always-flat agent
scores 0.0 and can "beat" a random policy whose net-of-cost Sharpe is negative — a PASS that means
"learned to sit still." And an always-long agent trivially beats random while having learned nothing.

**GATE-LEARN — the primary pass/fail.** ALL of:
1. `K >= 8` folds and `n_seeds >= 5`;
2. `mean(d) > 0`;
3. `t = mean(d) / clustered_standard_error({fold_k: [d[k,s] for s in seeds]}) >= t_crit(G−1, 0.975)`;
4. fold-level sign-flip permutation test (10,000 draws, flipping a whole fold's residuals together)
   gives `p < 0.05`;
5. the agent beats its fold's random mean in `>= 60%` of folds.

**GATE-OVERFIT — a warning, always reported.** Replay the deterministic policy over its *own* train
slice; report `overfit_gap = mean(Sharpe_IS − Sharpe_OOS)` with the same fold-clustered SE. Flag when
`gap > 1.0` or `mean(IS) > 0.5 while mean(OOS) <= 0` — the exact inversion documented in
arXiv:2112.04755 ("All agents underperform the QQEW benchmark OOS, although A2C and PPO outperform
IS") and arXiv:2309.03202 (Q-Learning best IS, worst OOS).

**BUY-AND-HOLD is measured, NOT gated.** Report `mean(S_agent − S_bh)` with the same fold-clustered
SE and CI, and print the literal line `beats buy-and-hold: NO (CI includes 0)` unless the CI strictly
excludes zero.

**Multiple-testing hygiene.** The report carries `n_configurations_evaluated`, incremented in the
`user.action` audit payload on every eval run against the same test span. Past 5 looks it prints the
Deflated-Sharpe caveat. Freeze the final 2 years as a **look-once holdout**.

Printed block (always, even on FAIL):

```
POLICY                ann.ret%  sharpe  sortino  maxDD%  turnover  winrate  exposure  trades
ppo (5 seeds, IQM)      ...       ...     ...      ...      ...      ...      ...      ...
buy_and_hold            ...       ...     ...      ...      ...      ...     1.00       1
always_flat            0.00      0.00    0.00     0.00     0.00       -      0.00       0
random (30 seeds, IQM)  ...       ...     ...      ...      ...      ...      ...      ...
sma_cross(20,50)        ...       ...     ...      ...      ...      ...      ...      ...

GATE-NOTDEGENERATE  PASS         exposure=0.58  entropy=0.94  corr(bh)=0.41
GATE-LEARN          PASS/FAIL/VOID  mean(d)=+0.31  CR0 SE=0.12  t=2.6 (t_crit(16)=2.12)  perm p=0.019  K=17
GATE-OVERFIT        WARN         IS sharpe 1.84 vs OOS 0.22 (gap 1.62)
vs BUY-AND-HOLD     NO           mean diff -0.44 sharpe, 95% CI [-0.79, -0.09]
break-even cost     3.1 bps      |  configs evaluated: 3  |  parity ties: 0/1000
```

---

## 6. Risks and pitfalls

| # | Risk | Mitigation |
|---|---|---|
| **R1** | **Chasing alpha and corrupting the loop.** The most likely failure is seeing "agent ties buy-and-hold," treating it as a bug, and iterating until something "works." That process *is* backtest overfitting. | Fix the eval protocol and ship criteria **before** the first training run. Log every trial in `n_configurations_evaluated`. Deflated-Sharpe machinery only works if you know how many trials you ran. |
| **R2** | **Reporting a single seed.** Any report module that *can* emit a single-seed number eventually will. | The eval entry point **requires** a seed list and refuses to run with `n < 5`. |
| **R3** | **Normalization leakage.** Fitting scalers on the full series, or `VecNormalize` with `training=True` at eval, silently leaks future moments into every observation. Highest-probability leak in the whole design. | Fixed a-priori constants inside the env. `VecNormalize` banned. Test A catches the class. |
| **R4** | **`PriceSeries.price_at` left-edge forward-look** (`if index < 0: return points[0][1]`). | Integer index slicing only. Test E asserts `price_at` never appears in `rl/env.py`. |
| **R5** | **`known_at` model is an assumption.** Alpaca's docs never state begin-vs-end stamping. If they switch, the fixed offset becomes a 16-hour backward error and every internal test still passes. | One-off live check against a real SPY bar before any `--confirm`. Documented as verified-by-inference. Offset kept conservative (early closes make it *late*, not early). |
| **R6** | **Torch/CUDA reaching CI or Render.** SB3 `.zip` loading needs torch + SB3; Linux torch wheels are the CUDA build. | (a) `rl` as a PEP-621 optional group; (b) torch-free JSON inference artifact + parity test; (c) import-linter `forbidden` contract; (d) **verify the Render dashboard build command by hand** — no `render.yaml` in-repo; (e) document the CPU index URL for local Linux training. |
| **R7** | **`mypy --strict` friction.** `files = ["src/trading", "tests"]` type-checks `src/trading/rl/**` in CI. `strict` enables `disallow_subclassing_any` (→ `gym.Env` as `Any` breaks the env class), `warn_return_any` (→ every SB3 call boundary errors), and `warn_unused_ignores` (→ an inline ignore needed *without* the extra becomes an error *with* it). | gymnasium + numpy in `dev` (954 KB, no torch) so both are properly typed. `ignore_missing_imports` only for `stable_baselines3.*` / `torch.*`. Narrow module overrides for `trading.rl.train` / `trading.rl.export`. **Never** inline `# type: ignore`. Never add `src/trading/rl` to mypy `exclude`. |
| **R8** | **Zero-cost reward ⇒ meaningless agent.** With no cost the optimal policy churns on noise, the learning curve looks fantastic, and the result is nothing. PPO specifically is documented trading frequently with short holding periods (arXiv:2407.09557). | Non-zero `cost_bps` is a **correctness requirement**, not a refinement. Costs charged inside equity so there is no weight to zero out. `ZERO_COST` exists only as a labeled ablation. |
| **R9** | **Perfect-fill paper broker.** `HistoricalMarketData.get_quote` returns `bid == ask == last`; `PaperBroker` stamps `fees=0`; Alpaca paper simulates no slippage/fees. Three different numbers (env reward, replay P&L, live paper P&L). | Label replay output cost-free. Publish `break_even_cost_bps`. Report the cost sweep. Never present env episode return as expected live return. |
| **R10** | **Inference determinism drift.** SB3's `predict` defaults to `deterministic=False` — sampling would violate the `Strategy` protocol's determinism contract and make the event log irreproducible. | Argmax only, in pure Python. Pin the artifact by sha256 and record it in the `DECISION_MADE` rationale. |
| **R11** | **Version drift silently changing a saved policy.** SB3 `.zip` checkpoints are cloudpickle-based and not guaranteed portable across major torch versions. | Pin `gymnasium>=1.3,<2`, `stable-baselines3>=2.9,<3`, `torch>=2.8,<3`. Store exact versions in the artifact sidecar. The JSON inference artifact has no torch dependency at all. |
| **R12** | **Sleeve capital is NOT isolated.** `allocated_capital` is display-only; the risk gate sizes against `account.equity` (the whole shared book) via one global `RiskGate`. | **Blocking prerequisite:** small fixed `rl_position_shares: Decimal = Decimal("1")` inside the strategy. The one risk here with real money consequences. |
| **R13** | **Strategies-layer import trap.** `trading.strategies` sits below `trading.application`; the RL strategy must not import `trading.rl`, torch, or SB3 at module scope — including transitively. A lazy torch import inside `evaluate()` would also violate "deterministic and side-effect free." | The policy is a pure `trading.domain` value injected at construction. Enforced by the `forbidden` contract + a test asserting the strategy's import closure is torch-free. |
| **R14** | **Composition-root import safety.** `tests/unit/test_main_roster.py` and `tests/e2e/test_composition_default_off.py` build the full graph. | RL sleeve must construct with **no torch import, no network, and no artifact file on disk** (`rl_policy_path=None` → inert). `registry.create("rl_ppo", ...)` must be total and return the inert form. |
| **R15** | **The cron doesn't drive the RL sleeve.** Without `jobs/rl_cycle.py` the sleeve is wired and never fires. | M9 is not optional. |
| **R16** | **`NoRealQuoteError` silently skips the symbol.** With `enable_live_quotes` off, SPY is skipped — no decision, no order, no error. | Expect "wired but does nothing" as the first observed behavior. `rl_cycle` dry-run prints the quote status explicitly. |
| **R17** | **Sample reuse / non-stationarity.** ~1,760 bars with 252-step episodes gives ~1,500 overlapping start indices; PPO sees the same regimes thousands of times. Any 7-year window is dominated by one or two regimes. arXiv:2309.03202's COVID result is the direct precedent. | Not fixable within the locked scope. Date split, purge, embargo, multi-seed variance, and honest reporting. **Don't promise robustness; promise measurement.** |
| **R18** | **The eval harness gives false rigor.** CR0 has no G/(G−1) correction; ticker clustering degenerates. | Cluster by fold, use `t(G−1)`, print `low_n` below K=8, and state the CR0 caveat in the report body. |
| **R19** | **SPY is a hindsight pick.** A US large-cap ETF whose history is a bull market, chosen with full knowledge of the outcome. | Right pick for a learning project (liquid, cheap, clean data). Named as a selection bias in the doc rather than left implicit. |
| **R20** | **Ruff/pre-commit friction.** DTZ bans naive datetimes (every bar timestamp, fold boundary, artifact field); T20 bans `print` outside the jobs' `# noqa: T201` escape hatch; RUF012 requires `metadata: ClassVar[...]`; B008 forbids `config: EnvConfig = EnvConfig()` in a signature. `scripts/check_memory_updated.sh` blocks commits touching `main.py`, `config.py`, `interface/app.py|schemas.py` without a same-commit MEMORY.md entry (adding a sleeve also changes `GET /sleeves` output — an API-surface change). | Budget for it. Don't reach for `--no-verify`. |

---

## 7. Milestone roadmap

Ordered for **time-to-demoable**, with the non-negotiable constraint that env + baselines + leakage
tests come before any agent. A demo built on a leaky env is worse than no demo — a reviewer will find
it.

Note the shape: **M2–M4 (env + leakage + baselines) are ~40% of the work and the agent itself (M5)
is the smallest milestone in the plan.** That ratio is the point.

| # | Milestone | New LOC (prod / test) | Effort | Demoable after? |
|---|---|---|---|---|
| **M0** | Dependency isolation + package skeleton | 40 config / 0 | 1-2 h | — |
| **M1** | SPY bar dataset job (`jobs/rl_dataset.py`) + `application/rl_dataset.py` | 200 / 130 | ½ day | — |
| **M2** | The environment (`rl/bars.py`, `rl/costs.py`, `rl/rewards.py`, `domain/rl_features.py`, `rl/env.py`) | 400 / 120 | 1.5 days | — |
| **M3** | **Leakage test suite** (incl. the `_LeakyEnv` negative control) | 10 / 200 | ½ day | ✅ **demo screen 1** |
| **M4** | Baselines + scoring (`strategies/rl_baselines.py`, `rl/scoring.py`, 3 metrics fns) | 200 / 140 | ½ day | ✅ baseline table |
| **M5** | The agent (`rl/train.py` + `jobs/rl_train.py`) — deliberately thin | 130 / 80 | 3-4 h | ✅ training curve |
| **M6** | Walk-forward eval + gates (`application/rl_evaluation.py`, `jobs/rl_eval.py`) | 320 / 200 | 1 day | ✅ **demo screen 2 — the headline** |
| **M7** | Torch-free export + parity (`rl/export.py`, `domain/rl_policy.py`, `infrastructure/rl/policy_store.py`, `jobs/rl_export.py`) | 220 / 100 | ½ day | ✅ isolation proof |
| **M8** | Strategy + sleeve wiring (`strategies/rl_policy.py`, `application/rl_observation.py`, main/config/AppDeps) | 150 / 120 | ½ day | ✅ `GET /sleeves` |
| **M9** | RL cycle driver (`jobs/rl_cycle.py`) | 150 / 130 | ½ day | ✅ **demo screen 3 — it trades** |
| **M10** | Decision explainability (obs + action + policy hash into `DECISION_MADE`) | 60 / 40 | 2 h | ✅ console attribution |
| **M11** *[O]* | Weekly scoring cron (live sleeve equity vs SPY) | 80 / 60 | ½ day | — |
| **M12** *[O]* | The tweaking surface (reward variants, cost sweep, lag, hyperparameters, 2nd algorithm) | — | ongoing | — |

**Minimum path to a full demo: M0 → M10.** ~5-7 focused days, ~1,900 LOC product + ~1,400 LOC tests.

---

## 8. Demo script (under 5 minutes)

**Pre-bake before the meeting:** run M1 and M5–M7 once and commit the artifacts
(`data/rl/spy_daily.json`, `artifacts/rl/spy_ppo_*.zip`, `artifacts/rl/spy_ppo_v1.json`,
`artifacts/rl/rl_eval.json`). The live demo replays and inspects; it does not train from scratch.

**Screen 0 — "the ML deps are not in the app" (20 s).**
```bash
cd backend && pip list | grep -Ei 'torch|stable-baselines3' ; lint-imports
```
Show: torch is absent from the `.[dev]` env, and `lint-imports` passes with the contract
`Heavy ML deps never enter the app import graph` listed. Then flip it: add `import trading.rl` to a
scratch copy of `main.py`, rerun, show it **fail**. *That contract is machine-checked, not a promise.*

**Screen 1 — "there is no lookahead, and here's the proof" (45 s).**
```bash
pytest tests/unit/rl/ -v -k "leakage or pit"
```
Walk the names out loud: `test_observation_is_invariant_to_every_future_bar[nan-120]`,
`test_every_bar_read_was_public_at_the_decision_time`,
`test_known_at_is_after_the_session_close_under_both_dst_regimes`, and — the one to point at —
`test_leakage_detector_actually_detects_leakage`. *"The last one proves the other tests can fail."*

**Screen 2 — the honest evaluation table (90 s). This is the headline.**
```bash
python -m trading.jobs.rl_eval \
    --dataset data/rl/spy_daily.json --models artifacts/rl/ \
    --seeds 0,1,2,3,4 --random-seeds 30 --cost-bps 0,2,5 --out artifacts/rl/rl_eval.json
```
Point at, in order: K folds, the four baseline rows, the random-policy IQM band, the cost sweep,
`break_even_cost_bps`, the fold-clustered SE with its `t(G−1)` critical value, and the literal
`beats buy-and-hold: NO (CI includes 0)` line. Say out loud: *"That's the expected result. With 7
years of daily bars the SE on an annualized Sharpe is 0.38 — you'd need ~40 years to resolve a 0.3
Sharpe gap. The deliverable is the measurement, not the return."*

**Screen 3 — torch-free inference parity (30 s).**
```bash
pytest tests/unit/rl/test_export_parity.py -v     # needs .[rl]; run in the training venv
python -c "import json,pathlib; \
  d=json.loads(pathlib.Path('artifacts/rl/spy_ppo_v1.json').read_text()); \
  print(d['policy_id'], d['schema_version'], [len(l['weight']) for l in d['hidden']])"
```
*"1,000 random observations, argmax in 40 lines of stdlib Python, identical to
`PPO.predict(deterministic=True)` every time. That JSON is what ships."*

**Screen 4 — it actually trades, in the real system (90 s).**
```bash
export RL_POLICY_PATH=artifacts/rl/spy_ppo_v1.json ENABLE_HISTORICAL_DATA=true
python -m trading.jobs.rl_cycle                  # DRY-RUN: prints obs vector, action, target position
python -m trading.jobs.rl_cycle --confirm        # places into the paper book, tagged sleeve-rl
```
Then the console:
- **Sleeves** screen → `sleeve-rl` alongside `sleeve-political`, sharing one Alpaca paper book.
- **Activity / Decisions** → the `DECISION_MADE` event with `rationale = "rl_ppo
  SPY-ppo-20260731-a1b2c3d4 action=buy obs=[...]"` and `market_context` carrying the quote.
- **Risk** → `RISK_EVALUATED` immediately after, showing the gate ran before the order.
- Per-sleeve P&L attribution updating.

*"Same risk gate, same event log, same reconciliation as the strategy that's been running for
months. The RL sleeve is one tuple in `sleeve_specs`."*

**Closing line (15 s):** *"The interesting engineering here isn't the PPO call — that's twelve lines
of stable-baselines3. It's everything that makes the number believable: the point-in-time bar model,
the leakage test with its own negative control, the walk-forward folds with purge, the matched random
null, costs charged inside the equity so there's no shaping weight to fudge, and a machine-checked
contract that keeps 900 MB of PyTorch out of the production import graph."*

---

## 9. Non-goals and stop conditions

### Explicit non-goals

1. **Beating buy-and-hold.** Out of scope, will not be claimed, and structurally not resolvable with
   this much data.
2. **Live trading.** `ALLOW_LIVE` stays off for this sleeve indefinitely. Paper only.
3. **Multi-asset, continuous actions, position sizing as an action, shorting.** Shorting is
   *impossible* (`TargetPosition` rejects negative quantity). The rest is deliberate scope control.
4. **Intraday bars, OHLCV features, order-book data.** `PriceSeries` carries **close only**
   (`alpaca_history.py:159`); OHLCV would require extending the adapter. Close-only is the honest
   cheap path for v1.
5. **A second backtest simulator.** Reuse `domain/metrics.py` and, for the risk-path replay,
   `InMemoryBacktestEnvFactory`.
6. **Hyperparameter search.** SB3 defaults plus the three documented deviations (§4.9). Any sweep must
   increment `n_configurations_evaluated` and is subject to the Deflated-Sharpe caveat.
7. **Per-sleeve risk limits / true capital isolation.** A separate, scoped change to the single global
   `RiskGate`.
8. **`gym-anytrading` / `FinRL` as dependencies.** Read gym-anytrading's `TradingEnv` as a reference;
   write our own.

### Stop conditions — stop and reassess if any of these hit

| Trigger | Action |
|---|---|
| The leakage suite's negative control (`_LeakyEnv`) **passes** the invariance check | STOP. The detector has no power. Fix the test before writing another line of RL code. |
| `assert_parity` disagrees on any observation with a top-2 logit margin `> 1e-4` | STOP. Do not write the artifact. The torch-free path is not equivalent; investigate `net_arch` / activation / features-extractor assumptions. |
| `K < 8` folds available from the dataset | STOP gating. Report `low_n`, print no PASS, and either widen the history or shrink `train_bars` — do not slice test windows thinner. |
| `risk_clipped_fraction > 0.10` | STOP the comparison. You are measuring the risk gate. Lower `rl_position_shares` and rerun. |
| `n_configurations_evaluated > 10` against the same test span | STOP tuning. Freeze the configuration, report the DSR caveat prominently, and move to the look-once holdout. |
| GATE-NOTDEGENERATE fails on every seed (always-long or always-flat) | Do **not** "fix" it with reward shaping. Record it as the expected outcome (§0.3), then optionally use `invalid_action_penalty` / `ent_coef` as *labeled experiments* with the gate re-run each time. |
| The env's cumulative reward and the `Backtest` equity curve diverge by more than the modeled cost | STOP. One of the two is wrong. This divergence is a designed bug detector. |
| A live Alpaca SPY bar's `t` does not match the session-open model in §4.2 | STOP all `--confirm` runs. The entire `known_at` model — and therefore every PIT guarantee — is invalid until re-derived. |
| Any RL code appears in `trading.{domain,application,strategies,interface,infrastructure,main}`'s import closure with torch/SB3 behind it | STOP. `lint-imports` should have caught it; if it didn't, the contract is misconfigured. |

### The success condition, restated

**Success is a report that says "indistinguishable from buy-and-hold after costs, with these
confidence intervals, on these folds, across these seeds" — and is right.** Predicting the failure
modes *in advance* (converges to always-buy; OOS collapse after an unseen regime; huge seed spread;
profit vanishing at 5 bps; train-best being test-worst) is what turns each one from a disappointment
into a confirmed lesson. That is the actual deliverable.

---

## 10. Source index

**Efficacy / results**
- Khare, Martheswaran & Dassanaike-Perera, *Evaluation of RL Techniques for Trading on a Diverse Portfolio* (SPY, tabular RL, COVID in/out-of-train) — <https://arxiv.org/abs/2309.03202>
- Zhang, Zohren & Roberts, *Deep Reinforcement Learning for Trading*, JFDS 2(2):25–40, 2020 — <https://arxiv.org/abs/1911.10107>
- Gort et al., *DRL for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting* (PBO test) — <https://arxiv.org/pdf/2209.05559>
- Mohammadshafie et al., *Deep RL Strategies in Finance* (PPO trades frequently, short holds) — <https://arxiv.org/abs/2407.09557>
- *Single stock trading with deep RL: A comparative study*, ICMLC 2022 (A2C/PPO underperform benchmark) — <https://dl.acm.org/doi/10.1145/3529836.3529857>
- Portfolio DRL IS/OOS inversion — <https://arxiv.org/pdf/2112.04755>

**Statistics / methodology**
- Lo, *The Statistics of Sharpe Ratios*, FAJ 58(4) 2002, eq. 9 + Table 1 — <https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf>
- Henderson et al., *Deep Reinforcement Learning that Matters* — <https://arxiv.org/abs/1709.06560>
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (IQM, stratified bootstrap, `rliable`) — <https://arxiv.org/abs/2108.13264>
- Bailey & López de Prado, *The Deflated Sharpe Ratio* — <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Purged / combinatorial purged cross-validation — <https://en.wikipedia.org/wiki/Purged_cross-validation>
- Seed-variance illustration (AlphaGen pure-PPO factor-mining baseline, 3 seeds, 91-day holdout; preprint, not peer reviewed) — <https://arxiv.org/abs/2606.29194>

**Reward design**
- Moody & Saffell, *Reinforcement Learning for Trading*, NeurIPS 1998 (additive profit eq. 1; differential Sharpe eqs. 4–5) — <https://proceedings.neurips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf>

**Libraries (all verified 2026-07-31)**
- <https://pypi.org/pypi/stable-baselines3/json> · <https://pypi.org/pypi/gymnasium/json> · <https://pypi.org/pypi/torch/json> · <https://pypi.org/pypi/numpy/json>
- <https://gymnasium.farama.org/api/env/> · <https://gymnasium.farama.org/introduction/create_custom_env/>
- <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html> · `.../guide/custom_env.html` · `.../guide/export.html` · `.../guide/save_format.html` · `.../common/env_checker.html` · `.../common/evaluation.html` · `.../common/monitor.html` · `.../guide/callbacks.html` · `.../guide/rl_tips.html`
- SB3 source (default `net_arch`, `MlpExtractor`, `FlattenExtractor`, `CategoricalDistribution.mode()`) — `stable_baselines3/common/policies.py`, `.../torch_layers.py` on master
- <https://pypi.org/pypi/gym-anytrading/json> (2.0.0, 2023-08-27) · <https://api.github.com/repos/AminHP/gym-anytrading> (last push 2024-03-14) · <https://pypi.org/pypi/finrl/json> (0.3.7, 2024-04-12)
- import-linter `forbidden` contract + `include_external_packages` — <https://import-linter.readthedocs.io/en/v2.9/contract_types/forbidden/>
- mypy strict flags — <https://mypy.readthedocs.io/en/stable/running_mypy.html>

**Data / costs**
- Alpaca bars endpoint (`adjustment` param, default `raw`; `t` semantics unstated) — <https://docs.alpaca.markets/us/reference/stockbars>
- Alpaca data plans ("7+ years" on both tiers) — <https://alpaca.markets/data> · <https://docs.alpaca.markets/us/docs/market-data-faq>
- Alpaca paper trading (no impact / latency slippage / fees; ~10% random partials) — <https://docs.alpaca.markets/us/docs/paper-trading>
- SSGA, SPY spread ($0.01 on $650.34, as of 2026-03-31) — <https://www.ssga.com/us/en/intermediary/insights/comparing-spy-and-splg-two-spdr-etfs-for-s-p-500-exposure>
- SEC Fee Rate Advisory FY2026 (§31: $20.60/$1M for sales on/after 2026-04-04) — <https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2>

---

## Change log
- **2026-07-31** — Initial program document. Locked: gymnasium 1.3 + SB3 2.9 PPO, single-ticker SPY
  `Discrete(3)`, optional `rl` dep group with a machine-checked `forbidden` contract, torch-free
  JSON inference artifact with a parity gate, walk-forward evaluation with fold-clustered CR0 SEs,
  and explicit non-goals. Corrected from earlier drafts: Alpaca history depth is **7+ years on both
  plans** (not 5 on free), giving SE(annualized Sharpe) ≈ **0.38**, not 0.45.
