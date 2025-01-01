import pytest

from optionlab.models import Inputs, Outputs, BlackScholesModelInputs
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
    "rho": [0.0, -0.04506390742751745],
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
    "rho": [0.08536880237502181, -0.07509774107468528],
}


def test_black_scholes():
    stock_price = 100.0
    strike = 105.0
    interest_rate = 1.0
    dividend_yield = 0.0
    volatility = 20.0
    days_to_maturity = 60

    interest_rate = interest_rate / 100
    dividend_yield = dividend_yield / 100
    volatility = volatility / 100
    time_to_maturity = days_to_maturity / 365

    bs = get_bs_info(
        stock_price, strike, interest_rate, volatility, time_to_maturity, dividend_yield
    )

    assert bs.call_price == 1.44
    assert bs.call_delta == 0.2942972000055033
    assert bs.call_theta == -8.780589609657586
    assert bs.call_rho == 0.04600635174517672
    assert bs.call_itm_prob == 0.2669832523577367
    assert bs.put_price == 6.27
    assert bs.put_delta == -0.7057027999944967
    assert bs.put_theta == -7.732314219179215
    assert bs.put_rho == -0.12631289052524033
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
            "days_to_target_date": 24,  # 32 days minus 9 non-business days plus 1 to consider the expiration date
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
        "rho": [0.0, -0.04506390742751745],
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

    assert outputs.model_dump(exclude={"data", "inputs"}, exclude_none=True) == {
        "probability_of_profit": 0.679058174271921,
        "profit_ranges": [(156.6, float("inf"))],
        "per_leg_cost": [-15899.0, -750.0, 990.0],
        "strategy_cost": -15659.0,
        "minimum_return_in_the_domain": -8760.000000000002,
        "maximum_return_in_the_domain": 11740.0,
        "implied_volatility": [0.0, 0.494, 0.482],
        "in_the_money_probability": [1.0, 0.54558925139931, 0.465831136209786],
        "delta": [1.0, 0.6039490632362865, -0.525237550169406],
        "gamma": [0.0, 0.015297136732317718, 0.015806160944019643],
        "theta": [0.0, -0.21821351060901806, 0.22301627833773927],
        "vega": [0.0, 0.20095091693287098, 0.20763771616023433],
        "rho": [0.0, 0.08536880237502181, -0.07509774107468528],
    }


def test_run_with_mc_array(nvidia):
    arr = create_price_samples(
        inputs=BlackScholesModelInputs(
            stock_price=168.99,
            volatility=0.483,
            interest_rate=0.045,
            years_to_target_date=24 / 365,
        ),
        seed=0,
    )

    inputs = Inputs.model_validate(
        nvidia
        | {
            "distribution": "array",
            "array": arr,
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
            "probability_of_profit": 0.56541,
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
            "rho": [0.0, -0.04506390742751745],
            "average_profit_from_mc": 1356.3702804556585,
            "average_loss_from_mc": -1407.9604829624866,
            "probability_of_profit_from_mc": 0.56564,
        },
        rel=0.05,
    )


# TODO: Reimplement compute expectation
# def test_100_itm_with_compute_expectation(nvidia):
#     inputs = Inputs.model_validate(
#         nvidia
#         | {
#             "compute_expectation": True,
#             # The covered call strategy is defined
#             "strategy": [
#                 {
#                     "type": "call",
#                     "strike": 165.0,
#                     "premium": 12.65,
#                     "n": 100,
#                     "action": "buy",
#                     "prev_pos": 7.5,
#                     "expiration": nvidia["target_date"],
#                 },
#                 {
#                     "type": "call",
#                     "strike": 170.0,
#                     "premium": 9.9,
#                     "n": 100,
#                     "action": "sell",
#                     "expiration": nvidia["target_date"],
#                 },
#             ],
#         }
#     )

#     outputs = run_strategy(inputs)

#     assert outputs.model_dump(
#         exclude={"data", "inputs"}, exclude_none=True
#     ) == pytest.approx(
#         PROB_100_ITM_RESULT
#         | {
#             "average_profit_from_mc": 492.7834646111533,
#             "average_loss_from_mc": 0.0,
#             "probability_of_profit_from_mc": 1.0,
#         },
#         rel=0.01,
#     )

# TODO: distribution='black-scholes' now is the same as distribution='normal'
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
            "mu": -0.07,
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
        COVERED_CALL_RESULT | {"probability_of_profit": 0.577830366334525}
    )


def test_calendar_spread():
    stock_price = 127.14  # Apple stock
    volatility = 0.427
    start_date = "2021-01-18"
    target_date = "2021-01-29"
    interest_rate = 0.0009
    min_stock = stock_price - round(stock_price * 0.5, 2)
    max_stock = stock_price + round(stock_price * 0.5, 2)
    strategy = [
        {
            "type": "call",
            "strike": 127.00,
            "premium": 4.60,
            "n": 1000,
            "action": "sell",
        },
        {
            "type": "call",
            "strike": 127.00,
            "premium": 5.90,
            "n": 1000,
            "action": "buy",
            "expiration": "2021-02-12",
        },
    ]

    inputs = {
        "stock_price": stock_price,
        "start_date": start_date,
        "target_date": target_date,
        "volatility": volatility,
        "interest_rate": interest_rate,
        "min_stock": min_stock,
        "max_stock": max_stock,
        "strategy": strategy,
    }

    outputs = run_strategy(inputs)

    assert outputs.model_dump(exclude={"data", "inputs"}, exclude_none=True) == {
        "probability_of_profit": 0.5991118190201975,
        "profit_ranges": [(118.87, 136.15)],
        "per_leg_cost": [4600.0, -5900.0],
        "strategy_cost": -1300.0,
        "minimum_return_in_the_domain": -1300.0000000000146,
        "maximum_return_in_the_domain": 3009.999999999999,
        "implied_volatility": [0.47300000000000003, 0.419],
        "in_the_money_probability": [0.4895105709759477, 0.4805997906939539],
        "delta": [-0.5216914758915705, 0.5273457614638198],
        "gamma": [0.03882722919950356, 0.02669940508461828],
        "theta": [0.22727438444823292, -0.15634971608107964],
        "vega": [0.09571294014902997, 0.1389462831961853],
        "rho": [-0.022202087247849632, 0.046016214466188525],
    }
