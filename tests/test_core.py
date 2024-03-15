import pytest

from optionlab.models import Inputs, Outputs
from optionlab.engine import StrategyEngine
from optionlab.support import create_price_samples

COVERED_CALL_RESULT = {
    "probability_of_profit": 0.5489826392738772,
    "profit_ranges": [(164.9, float("inf"))],
    "per_leg_cost": [-16899.0, 409.99999999999994],
    "strategy_cost": -16489.0,
    "minimum_return_in_the_domain": -9590.000000000002,
    "maximum_return_in_the_domain": 2011.0,
    "implied_volatility": [0.0, 0.466],
    "in_the_money_probability": [1.0, 0.2529827985340476],
    "delta": [1.0, -0.30180572515271814],
    "gamma": [0.0, 0.01413835937607837],
    "theta": [0.0, 0.19521264859629808],
    "vega": [0.0, 0.1779899391089498],
}

PROB_100_ITM_RESULT = {
    "probability_of_profit": 1.0,
    "profit_ranges": [(0.0, float("inf"))],
    "per_leg_cost": [-750.0, 990.0],
    "strategy_cost": 240.0,
    "minimum_return_in_the_domain": 240.0,
    "maximum_return_in_the_domain": 740.0000000000018,
    "implied_volatility": [0.505, 0.493],
    "in_the_money_probability": [0.547337257503663, 0.4658724723221915],
    "delta": [0.6044395589860037, -0.5240293090819207],
    "gamma": [0.015620889396345561, 0.016149144698391314],
    "theta": [-0.22254722153197432, 0.22755381063645636],
    "vega": [0.19665373318968424, 0.20330401888012928],
}


def test_covered_call(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-2-mixing-options-and-stocks-1e9f59f388f

    inputs = Inputs.model_validate(
        nvidia
        | {
            # The covered call strategy is defined
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy"},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(exclude_none=True) == pytest.approx(COVERED_CALL_RESULT)


def test_covered_call_w_days_to_target(nvidia):
    inputs = Inputs.model_validate(
        nvidia
        | {
            "start_date": None,
            "target_date": None,
            "days_to_target_date": 23,  # 32 days minus 9 non-business days
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy"},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(exclude_none=True) == pytest.approx(COVERED_CALL_RESULT)


def test_covered_call_w_prev_position(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-2-mixing-options-and-stocks-1e9f59f388f

    inputs = Inputs.model_validate(
        nvidia
        | {
            # The covered call strategy is defined
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy", "prev_pos": 158.99},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ]
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert outputs.model_dump(exclude_none=True) == {
        "probability_of_profit": 0.7094641281976972,
        "profit_ranges": [(154.9, float("inf"))],
        "per_leg_cost": [-15899.0, 409.99999999999994],
        "strategy_cost": -15489.0,
        "minimum_return_in_the_domain": -8590.000000000002,
        "maximum_return_in_the_domain": 3011.0,
        "implied_volatility": [0.0, 0.466],
        "in_the_money_probability": [1.0, 0.2529827985340476],
        "delta": [1.0, -0.30180572515271814],
        "gamma": [0.0, 0.01413835937607837],
        "theta": [0.0, 0.19521264859629808],
        "vega": [0.0, 0.1779899391089498],
    }


def test_100_perc_itm(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-3-a-trade-with-100-probability-of-profit-886e934addbf

    inputs = Inputs.model_validate(
        nvidia
        | {
            # The covered call strategy is defined
            "strategy": [
                {
                    "type": "call",
                    "strike": 165.0,
                    "premium": 12.65,
                    "n": 100,
                    "action": "buy",
                    "prev_pos": 7.5,
                    "expiration": nvidia["target_date"],
                },
                {
                    "type": "call",
                    "strike": 170.0,
                    "premium": 9.9,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ]
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert outputs.model_dump(exclude_none=True) == pytest.approx(PROB_100_ITM_RESULT)


def test_3_legs(nvidia):

    inputs = Inputs.model_validate(
        nvidia
        | {
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy", "prev_pos": 158.99},
                {
                    "type": "call",
                    "strike": 165.0,
                    "premium": 12.65,
                    "n": 100,
                    "action": "buy",
                    "prev_pos": 7.5,
                    "expiration": nvidia["target_date"],
                },
                {
                    "type": "call",
                    "strike": 170.0,
                    "premium": 9.9,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ]
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert len(outputs.per_leg_cost) == 3
    assert len(outputs.delta) == 3


def test_run_with_mc_array(nvidia):
    array_prices = create_price_samples(168.99, 0.483, 23 / 365, 0.045)

    inputs = Inputs.model_validate(
        nvidia
        | {
            "distribution": "array",
            "array_prices": array_prices,
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy"},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    assert outputs.model_dump(exclude_none=True) == pytest.approx(
        {
            "probability_of_profit": 0.56679,
            "profit_ranges": [(164.9, float("inf"))],
            "per_leg_cost": [-16899.0, 409.99999999999994],
            "strategy_cost": -16489.0,
            "minimum_return_in_the_domain": -9590.000000000002,
            "maximum_return_in_the_domain": 2011.0,
            "implied_volatility": [0.0, 0.466],
            "in_the_money_probability": [1.0, 0.2529827985340476],
            "delta": [1.0, -0.30180572515271814],
            "gamma": [0.0, 0.01413835937607837],
            "theta": [0.0, 0.19521264859629808],
            "vega": [0.0, 0.1779899391089498],
            "average_profit_from_mc": 1348.2950516297647,
            "average_loss_from_mc": -1388.1981940251862,
            "probability_of_profit_from_mc": 0.56703,
        },
        rel=0.05,
    )


def test_100_itm_with_compute_expectation(nvidia):
    inputs = Inputs.model_validate(
        nvidia
        | {
            "compute_expectation": True,
            # The covered call strategy is defined
            "strategy": [
                {
                    "type": "call",
                    "strike": 165.0,
                    "premium": 12.65,
                    "n": 100,
                    "action": "buy",
                    "prev_pos": 7.5,
                    "expiration": nvidia["target_date"],
                },
                {
                    "type": "call",
                    "strike": 170.0,
                    "premium": 9.9,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    assert outputs.model_dump(exclude_none=True) == pytest.approx(
        PROB_100_ITM_RESULT
        | {
            "average_profit_from_mc": 493.3532975418169,
            "average_loss_from_mc": 0.0,
            "probability_of_profit_from_mc": 1.0,
        },
        rel=0.01,
    )


def test_covered_call_w_normal_distribution(nvidia):

    inputs = Inputs.model_validate(
        nvidia
        | {
            "distribution": "normal",
            # The covered call strategy is defined
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy"},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(exclude_none=True) == pytest.approx(
        COVERED_CALL_RESULT | {"probability_of_profit": 0.5666705670736036}
    )


def test_covered_call_w_laplace_distribution(nvidia):

    inputs = Inputs.model_validate(
        nvidia
        | {
            "distribution": "laplace",
            # The covered call strategy is defined
            "strategy": [
                {"type": "stock", "n": 100, "action": "buy"},
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                    "expiration": nvidia["target_date"],
                },
            ],
        }
    )

    st = StrategyEngine(inputs)
    outputs = st.run()

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(exclude_none=True) == pytest.approx(
        COVERED_CALL_RESULT | {"probability_of_profit": 0.60568262830598}
    )
