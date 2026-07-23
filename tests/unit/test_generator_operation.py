import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.utilities.generator_operation import configure_nuclear


@pytest.fixture
def test_network() -> pypsa.Network:
    """Create a small network containing nuclear and gas generators."""
    network = pypsa.Network()
    network.add("Bus", "bus")

    network.add(
        "Generator",
        "nuclear_extendable",
        bus="bus",
        carrier="nuclear",
        p_nom=1_000.0,
        p_nom_extendable=True,
        committable=False,
        p_min_pu=0.0,
    )

    network.add(
        "Generator",
        "nuclear_committable",
        bus="bus",
        carrier="Nuclear",
        p_nom=1_000.0,
        p_nom_extendable=False,
        committable=True,
        p_min_pu=0.0,
    )

    network.add(
        "Generator",
        "gas",
        bus="bus",
        carrier="gas",
        p_nom=1_000.0,
        p_min_pu=0.1,
        ramp_limit_up=0.5,
        ramp_limit_down=0.4,
    )

    return network


def test_configure_nuclear_applies_constraints(
        test_network: pypsa.Network,
) -> None:
    """Nuclear generators should receive the configured constraints."""
    result = configure_nuclear(
        network=test_network,
        min_output=0.2,
        ramp_limit=0.025,
    )

    nuclear_names = [
        "nuclear_extendable",
        "nuclear_committable",
    ]

    np.testing.assert_allclose(
        result.generators.loc[nuclear_names, "p_min_pu"],
        0.2,
    )
    np.testing.assert_allclose(
        result.generators.loc[nuclear_names, "ramp_limit_up"],
        0.025,
    )
    np.testing.assert_allclose(
        result.generators.loc[nuclear_names, "ramp_limit_down"],
        0.025,
    )

    assert result is test_network


def test_configure_nuclear_preserves_other_generators(
        test_network: pypsa.Network,
) -> None:
    """Non-nuclear generators should remain unchanged."""
    gas_before = test_network.generators.loc["gas"].copy()

    configure_nuclear(
        network=test_network,
        min_output=0.2,
        ramp_limit=0.025,
    )

    pd.testing.assert_series_equal(
        test_network.generators.loc["gas"],
        gas_before,
    )


def test_configure_nuclear_preserves_model_choices(
        test_network: pypsa.Network,
) -> None:
    """Capacity-expansion and commitment choices should be preserved."""
    extendable_before = test_network.generators[
        "p_nom_extendable"
    ].copy()
    committable_before = test_network.generators[
        "committable"
    ].copy()

    configure_nuclear(
        network=test_network,
        min_output=0.2,
        ramp_limit=0.025,
    )

    pd.testing.assert_series_equal(
        test_network.generators["p_nom_extendable"],
        extendable_before,
    )
    pd.testing.assert_series_equal(
        test_network.generators["committable"],
        committable_before,
    )


@pytest.mark.parametrize(
    ("min_output", "ramp_limit"),
    [
        (-0.01, 0.1),
        (1.01, 0.1),
        (0.1, -0.01),
        (0.1, 1.01),
        (np.nan, 0.1),
        (0.1, np.nan),
        (np.inf, 0.1),
        (0.1, np.inf),
        ("invalid", 0.1),
        (0.1, "invalid"),
        (True, 0.1),
        (0.1, False),
    ],
)
def test_configure_nuclear_rejects_invalid_values(
        test_network: pypsa.Network,
        min_output: object,
        ramp_limit: object,
) -> None:
    """Invalid per-unit settings should raise a useful error."""
    with pytest.raises(ValueError):
        configure_nuclear(
            network=test_network,
            min_output=min_output,
            ramp_limit=ramp_limit,
        )


@pytest.mark.parametrize(
    ("min_output", "ramp_limit"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_configure_nuclear_accepts_boundary_values(
        test_network: pypsa.Network,
        min_output: float,
        ramp_limit: float,
) -> None:
    """Zero and one should be valid inclusive boundary values."""
    configure_nuclear(
        network=test_network,
        min_output=min_output,
        ramp_limit=ramp_limit,
    )


def test_configure_nuclear_requires_nuclear_generators() -> None:
    """Enabled nuclear configuration should not silently do nothing."""
    network = pypsa.Network()
    network.add("Bus", "bus")
    network.add(
        "Generator",
        "gas",
        bus="bus",
        carrier="gas",
        p_nom=1_000.0,
    )

    with pytest.raises(ValueError, match="nuclear generators"):
        configure_nuclear(
            network=network,
            min_output=0.2,
            ramp_limit=0.025,
        )
