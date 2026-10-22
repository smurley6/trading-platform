"""The env passes BOTH env checkers (task-rl-08 acceptance test 4; program doc T0 / §4.8 Test F).

``rl``-marked: the stable-baselines3 checker needs the optional ``[rl]`` extra and is skipped
without it (``pytest.importorskip``).
"""

import warnings

import pytest

from tests.unit.rl.conftest import make_env, synthetic_random_walk

pytestmark = pytest.mark.rl
sb3_env_checker = pytest.importorskip("stable_baselines3.common.env_checker")


def test_stable_baselines3_env_checker_passes_clean() -> None:
    env = make_env(synthetic_random_walk(400, seed=3), episode_length=252)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb3_env_checker.check_env(env, warn=True, skip_render_check=True)
    assert [str(w.message) for w in caught] == []


def test_gymnasium_env_checker_passes_clean() -> None:
    from gymnasium.utils.env_checker import check_env

    env = make_env(synthetic_random_walk(400, seed=3), episode_length=252)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env)
    # The only notice gymnasium emits is that an unregistered env (no gymnasium.make spec)
    # cannot be re-made in alternative render modes — this env declares none.
    assert all("not having a spec" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
