"""Configuration errors fail before an expensive evolutionary run begins."""

from dataclasses import FrozenInstanceError
import math

import pytest

from gp.config import GPConfig, load_config


def test_defaults_are_valid_and_round_trip():
    config = GPConfig()
    assert GPConfig.from_dict(config.to_dict()) == config
    assert config.max_depth == 7
    assert config.constant_range == (-5.0, 5.0)
    assert config.reproduction_probability == pytest.approx(0.05)
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


@pytest.mark.parametrize("selection", ["tournament", "roulette", "epsilon_lexicase"])
def test_supported_selection_methods(selection):
    assert GPConfig(selection=selection).selection == selection


def test_root_depth_zero_and_no_generations_are_valid():
    config = GPConfig(min_initial_depth=0, max_depth=0, generations=0, elite_size=0)
    assert config.max_depth == 0


def test_exclusive_operators_can_use_all_probability():
    assert GPConfig(pc=0.6, pm=0.4).reproduction_probability == pytest.approx(0.0)
    assert GPConfig(pc=0.0, pm=0.0).reproduction_probability == 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"population_size": 1},
        {"population_size": True},
        {"generations": -1},
        {"generations": 1.5},
        {"min_initial_depth": -1},
        {"min_initial_depth": 3, "max_depth": 2},
        {"max_depth": -1},
        {"tournament_size": 0},
        {"tournament_size": 101},
        {"elite_size": -1},
        {"elite_size": 100},
        {"selection": "unknown"},
        {"pc": -0.1},
        {"pm": 1.1},
        {"pc": 0.8, "pm": 0.3},
        {"pc": math.nan},
        {"pm": math.inf},
        {"pc": True},
        {"constant_min": 6.0},
        {"constant_max": math.inf},
        {"seed": -1},
        {"functions": ()},
        {"functions": "add"},
        {"functions": ("unknown",)},
        {"functions": ("add", "add")},
        {"functions": (1,)},
    ],
)
def test_invalid_configurations_are_rejected(changes):
    with pytest.raises(ValueError):
        GPConfig(**changes)


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="populaton_size"):
        GPConfig.from_dict({"populaton_size": 10})


@pytest.mark.parametrize("section", ["", "[gp]\n"])
def test_load_toml_root_or_gp_section(tmp_path, section):
    path = tmp_path / "run.toml"
    path.write_text(section + 'population_size = 20\nfunctions = ["add", "div"]\nseed = 42\n')
    config = load_config(path)
    assert config.population_size == 20
    assert config.functions == ("add", "div")
    assert config.seed == 42


def test_load_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("[gp]\npopulation = 20\n")
    with pytest.raises(ValueError, match="population"):
        load_config(path)


def test_load_config_rejects_non_mapping_gp(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("gp = 5\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
