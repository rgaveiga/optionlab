import pytest

from optionlab.models import Inputs, Outputs
from optionlab.engine import run_strategy
from optionlab.support import create_price_samples
from optionlab.black_scholes import get_bs_info

COVERED_CALL_RESULT = {
    "probability_of_profit": 0.5472008423945269,
    "profit_ranges": [(164.9, float("inf"))],
    "per_leg_cost": [-16899.0, 409.99999999999994],
    "strategy_cost": -16489.0,
    "minimum_return_in_the_domain": -9590.000000000002,
    "maximum_return_in_the_domain": 2011.0,
    "implied_volatility": [0.0, 0.456],
    "in_the_money_probability": [1.0, 0.256866624586934],
    "delta": [1.0, -0.30713817729665704],
    "gamma": [0.0, 0.013948977387090415],
    "theta": [0.0, 0.19283555235589467],
    "vega": [0.0, 0.1832408146218486],
}

PROB_100_ITM_RESULT = {
    "probability_of_profit": 1.0,
    "profit_ranges": [(0.0, float("inf"))],
    "per_leg_cost": [-750.0, 990.0],
    "strategy_cost": 240.0,
    "minimum_return_in_the_domain": 240.0,
    "maximum_return_in_the_domain": 740.0000000000018,
    "implied_volatility": [0.494, 0.482],
    "in_the_money_probability": [0.54558925139931, 0.465831136209786],
    "delta": [0.6039490632362865, -0.525237550169406],
    "gamma": [0.015297136732317718, 0.015806160944019643],
    "theta": [-0.21821351060901806, 0.22301627833773927],
    "vega": [0.20095091693287098, 0.20763771616023433],
}


def test_black_scholes():
    stockprice = 100.0
    strike = 105.0
    interestrate = 1.0
    dividendyield = 0.0
    volatility = 20.0
    days2maturity = 60

    interestrate = interestrate / 100
    dividendyield = dividendyield / 100
    volatility = volatility / 100
    time_to_maturity = days2maturity / 365

    bs = get_bs_info(
        stockprice, strike, interestrate, volatility, time_to_maturity, dividendyield
    )

    assert bs.call_price == 1.44
    assert bs.call_delta == 0.2942972000055033
    assert bs.call_theta == -8.780589609657586
    assert bs.call_itm_prob == 0.2669832523577367
    assert bs.put_price == 6.27
    assert bs.put_delta == -0.7057027999944967
    assert bs.put_theta == -7.732314219179215
    assert bs.put_itm_prob == 0.7330167476422633
    assert bs.gamma == 0.042503588182705464
    assert bs.vega == 0.13973782416231934


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

    outputs = run_strategy(inputs)

    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(COVERED_CALL_RESULT)


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

    outputs = run_strategy(inputs)

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(COVERED_CALL_RESULT)


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

    outputs = run_strategy(inputs)

    # Print useful information on screen
    assert outputs.model_dump(exclude={"data", "inputs"}, exclude_none=True) == {
        "probability_of_profit": 0.7048129541301167,
        "profit_ranges": [(154.9, float("inf"))],
        "per_leg_cost": [-15899.0, 409.99999999999994],
        "strategy_cost": -15489.0,
        "minimum_return_in_the_domain": -8590.000000000002,
        "maximum_return_in_the_domain": 3011.0,
        "implied_volatility": [0.0, 0.456],
        "in_the_money_probability": [1.0, 0.256866624586934],
        "delta": [1.0, -0.30713817729665704],
        "gamma": [0.0, 0.013948977387090415],
        "theta": [0.0, 0.19283555235589467],
        "vega": [0.0, 0.1832408146218486],
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

    outputs = run_strategy(inputs)

    # Print useful information on screen
    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(PROB_100_ITM_RESULT)


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

    outputs = run_strategy(inputs)

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

    outputs = run_strategy(inputs)

    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(
        {
            "probability_of_profit": 0.56141,
            "profit_ranges": [(164.9, float("inf"))],
            "per_leg_cost": [-16899.0, 409.99999999999994],
            "strategy_cost": -16489.0,
            "minimum_return_in_the_domain": -9590.000000000002,
            "maximum_return_in_the_domain": 2011.0,
            "implied_volatility": [0.0, 0.456],
            "in_the_money_probability": [1.0, 0.256866624586934],
            "delta": [1.0, -0.30713817729665704],
            "gamma": [0.0, 0.013948977387090415],
            "theta": [0.0, 0.19283555235589467],
            "vega": [0.0, 0.1832408146218486],
            "average_profit_from_mc": 1358.707606387012,
            "average_loss_from_mc": -1408.7310982891534,
            "probability_of_profit_from_mc": 0.5616,
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

    outputs = run_strategy(inputs)

    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(
        PROB_100_ITM_RESULT
        | {
            "average_profit_from_mc": 492.7834646111533,
            "average_loss_from_mc": 0.0,
            "probability_of_profit_from_mc": 1.0,
        },
        rel=0.01,
    )


# def test_covered_call_w_normal_distribution(nvidia):
#     inputs = Inputs.model_validate(
#         nvidia
#         | {
#             "distribution": "normal",
#             # The covered call strategy is defined
#             "strategy": [
#                 {"type": "stock", "n": 100, "action": "buy"},
#                 {
#                     "type": "call",
#                     "strike": 185.0,
#                     "premium": 4.1,
#                     "n": 100,
#                     "action": "sell",
#                     "expiration": nvidia["target_date"],
#                 },
#             ],
#         }
#     )

#     outputs = run_strategy(inputs)

#     # Print useful information on screen
#     assert isinstance(outputs, Outputs)
#     assert outputs.model_dump(
#         exclude={"data", "inputs"}, exclude_none=True
#     ) == pytest.approx(
#         COVERED_CALL_RESULT | {"probability_of_profit": 0.565279550918542}
#     )


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

    outputs = run_strategy(inputs)

    # Print useful information on screen
    assert isinstance(outputs, Outputs)
    assert outputs.model_dump(
        exclude={"data", "inputs"}, exclude_none=True
    ) == pytest.approx(
        COVERED_CALL_RESULT | {"probability_of_profit": 0.5772025728573296}
    )
