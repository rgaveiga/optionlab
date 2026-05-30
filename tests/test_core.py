import pytest

from optionlab import Inputs, run_strategy
from optionlab.black_scholes import get_bs_info
from optionlab.models import Outputs


OUTPUT_EXCLUDE_FIELDS = {"data", "inputs"}


COVERED_CALL_RESULT = {
    "probability_of_profit": 0.5472008423945267,
    "expected_profit_if_profitable": 1448.28,
    "expected_loss_if_unprofitable": -1703.74,
    "profit_ranges": [(164.9, float("inf"))],
    "per_leg_cost": [-16899.0, 409.99999999999994],
    "strategy_cost": -16489.0,
    "minimum_return_in_the_domain": -9590.000000000002,
    "maximum_return_in_the_domain": 2011.0,
    "implied_volatility": [0.0, 0.456],
    "in_the_money_probability": [1.0, 0.256866624586934],
    "probability_of_touch": [1.0, 0.5277250352054264],
    "delta": [1.0, -0.30713817729665704],
    "gamma": [0.0, 0.013948977387090415],
    "theta": [0.0, 0.19283555235589467],
    "vega": [0.0, 0.1832408146218486],
    "rho": [0.0, -0.04506390742751745],
}

PROB_100_ITM_RESULT = {
    "probability_of_profit": 1.0,
    "expected_profit_if_profitable": 492.57,
    "profit_ranges": [(0.0, float("inf"))],
    "per_leg_cost": [-750.0, 990.0],
    "strategy_cost": 240.0,
    "minimum_return_in_the_domain": 240.0,
    "maximum_return_in_the_domain": 740.0000000000018,
    "implied_volatility": [0.494, 0.483],
    "in_the_money_probability": [0.54558925139931, 0.465831136209786],
    "probability_of_touch": [1.0, 0.9661799112521838],
    "delta": [0.6039490632362865, -0.525237550169406],
    "gamma": [0.015297136732317718, 0.015806160944019643],
    "theta": [-0.21821351060901806, 0.22301627833773927],
    "vega": [0.20095091693287098, 0.20763771616023433],
    "rho": [0.08536880237502181, -0.07509774107468528],
}

NAKED_CALL = {
    "probability_of_profit": 0.8389215512144531,
    "expected_profit_if_profitable": 113.49,
    "expected_loss_if_unprofitable": -717.5,
    "profit_ranges": [(0.0, 176.14)],
    "per_leg_cost": [114.99999999999999],
    "strategy_cost": 114.99999999999999,
    "minimum_return_in_the_domain": -6991.999999999999,
    "maximum_return_in_the_domain": 114.99999999999999,
    "implied_volatility": [0.256],
    "in_the_money_probability": [0.1832371984432129],
    "probability_of_touch": [0.3741546603689868],
    "delta": [-0.20371918274704337],
    "gamma": [0.023104402361599465],
    "theta": [0.091289876347897],
    "vega": [0.12750177318341913],
    "rho": [-0.02417676577711979],
    "probability_of_profit_target": 0.8197909190785164,
    "profit_target_ranges": [(0.0, 175.15)],
    "probability_of_loss_limit": 0.14307836806156238,
    "loss_limit_ranges": [(177.15, float("inf"))],
}

BLACK_SCHOLES_EXPECTED = {
    "call_price": 1.4425226889011533,
    "call_delta": 0.2942972000055033,
    "call_theta": -8.780589609657586,
    "call_rho": 0.04600635174517672,
    "call_itm_prob": 0.2669832523577367,
    "call_prob_of_touch": 0.540374479063,
    "put_price": 6.270061736738214,
    "put_delta": -0.7057027999944967,
    "put_theta": -7.732314219179215,
    "put_rho": -0.12631289052524033,
    "put_itm_prob": 0.7330167476422633,
    "put_prob_of_touch": 1.0,
    "gamma": 0.042503588182705464,
    "vega": 0.13973782416231934,
}

COVERED_CALL_LEGS = [
    {"type": "stock", "n": 100, "action": "buy"},
    {
        "type": "call",
        "strike": 185.0,
        "premium": 4.1,
        "n": 100,
        "action": "sell",
    },
]


def run_validated_strategy(payload):
    outputs = run_strategy(Inputs.model_validate(payload))

    assert isinstance(outputs, Outputs)

    return outputs.model_dump(
        exclude=OUTPUT_EXCLUDE_FIELDS,
        exclude_none=True,
        exclude_defaults=True,
    )


def with_expiration(strategy, expiration):
    return [
        leg if leg["type"] == "stock" else leg | {"expiration": expiration}
        for leg in strategy
    ]


def test_black_scholes():
    bs = get_bs_info(
        s=100.0,
        x=105.0,
        r=1.0 / 100,
        vol=20.0 / 100,
        years_to_maturity=60 / 365,
        y=0.0,
    )

    assert bs.call_price == BLACK_SCHOLES_EXPECTED["call_price"]
    assert bs.call_delta == BLACK_SCHOLES_EXPECTED["call_delta"]
    assert bs.call_theta == BLACK_SCHOLES_EXPECTED["call_theta"]
    assert bs.call_rho == BLACK_SCHOLES_EXPECTED["call_rho"]
    assert bs.call_itm_prob == BLACK_SCHOLES_EXPECTED["call_itm_prob"]
    assert round(bs.call_prob_of_touch, 12) == BLACK_SCHOLES_EXPECTED["call_prob_of_touch"]
    assert bs.put_price == BLACK_SCHOLES_EXPECTED["put_price"]
    assert bs.put_delta == BLACK_SCHOLES_EXPECTED["put_delta"]
    assert bs.put_theta == BLACK_SCHOLES_EXPECTED["put_theta"]
    assert bs.put_rho == BLACK_SCHOLES_EXPECTED["put_rho"]
    assert bs.put_itm_prob == BLACK_SCHOLES_EXPECTED["put_itm_prob"]
    assert bs.put_prob_of_touch == BLACK_SCHOLES_EXPECTED["put_prob_of_touch"]
    assert bs.gamma == BLACK_SCHOLES_EXPECTED["gamma"]
    assert bs.vega == BLACK_SCHOLES_EXPECTED["vega"]


def test_covered_call(nvidia):
    payload = nvidia | {"strategy": with_expiration(COVERED_CALL_LEGS, nvidia["target_date"])}

    assert run_validated_strategy(payload) == pytest.approx(COVERED_CALL_RESULT)


def test_covered_call_w_days_to_target(nvidia):
    payload = nvidia | {
        "start_date": None,
        "target_date": None,
        "days_to_target_date": 24,  # 32 days minus 9 non-business days plus 1 to consider the expiration date
        "strategy": COVERED_CALL_LEGS,
    }

    assert run_validated_strategy(payload) == pytest.approx(COVERED_CALL_RESULT)


def test_covered_call_w_prev_position(nvidia):
    payload = nvidia | {
        "strategy": with_expiration(
            [
                {"type": "stock", "n": 100, "action": "buy", "prev_pos": 158.99},
                COVERED_CALL_LEGS[1],
            ],
            nvidia["target_date"],
        )
    }

    assert run_validated_strategy(payload) == {
        "probability_of_profit": 0.7048129541301169,
        "expected_profit_if_profitable": 2013.63,
        "expected_loss_if_unprofitable": -1350.06,
        "profit_ranges": [(154.9, float("inf"))],
        "per_leg_cost": [-15899.0, 409.99999999999994],
        "strategy_cost": -15489.0,
        "minimum_return_in_the_domain": -8590.000000000002,
        "maximum_return_in_the_domain": 3011.0,
        "implied_volatility": [0.0, 0.456],
        "in_the_money_probability": [1.0, 0.256866624586934],
        "probability_of_touch": [1.0, 0.5277250352054264],
        "delta": [1.0, -0.30713817729665704],
        "gamma": [0.0, 0.013948977387090415],
        "theta": [0.0, 0.19283555235589467],
        "vega": [0.0, 0.1832408146218486],
        "rho": [0.0, -0.04506390742751745],
    }


def test_100_perc_itm(nvidia):
    payload = nvidia | {
        "strategy": with_expiration(
            [
                {
                    "type": "call",
                    "strike": 165.0,
                    "premium": 12.65,
                    "n": 100,
                    "action": "buy",
                    "prev_pos": 7.5,
                },
                {
                    "type": "call",
                    "strike": 170.0,
                    "premium": 9.9,
                    "n": 100,
                    "action": "sell",
                },
            ],
            nvidia["target_date"],
        )
    }

    assert run_validated_strategy(payload) == pytest.approx(PROB_100_ITM_RESULT)


def test_naked_call():
    payload = {
        "stock_price": 164.04,
        "volatility": 0.272,
        "start_date": "2021-11-22",
        "target_date": "2021-12-17",
        "interest_rate": 0.0002,
        "min_stock": 82.02,
        "max_stock": 246.06,
        "profit_target": 100.0,
        "loss_limit": -100.0,
        "model": "black-scholes",
        "strategy": [
            {
                "type": "call",
                "strike": 175.00,
                "premium": 1.15,
                "n": 100,
                "action": "sell",
            }
        ],
    }

    assert run_validated_strategy(payload) == pytest.approx(NAKED_CALL)


def test_3_legs(nvidia):
    payload = nvidia | {
        "strategy": with_expiration(
            [
                {"type": "stock", "n": 100, "action": "buy", "prev_pos": 158.99},
                {
                    "type": "call",
                    "strike": 165.0,
                    "premium": 12.65,
                    "n": 100,
                    "action": "buy",
                    "prev_pos": 7.5,
                },
                {
                    "type": "call",
                    "strike": 170.0,
                    "premium": 9.9,
                    "n": 100,
                    "action": "sell",
                },
            ],
            nvidia["target_date"],
        )
    }

    assert run_validated_strategy(payload) == {
        "probability_of_profit": 0.6790581742719213,
        "expected_profit_if_profitable": 2956.8,
        "expected_loss_if_unprofitable": -1404.83,
        "profit_ranges": [(156.6, float("inf"))],
        "per_leg_cost": [-15899.0, -750.0, 990.0],
        "strategy_cost": -15659.0,
        "minimum_return_in_the_domain": -8760.000000000002,
        "maximum_return_in_the_domain": 11740.0,
        "implied_volatility": [0.0, 0.494, 0.483],
        "in_the_money_probability": [1.0, 0.54558925139931, 0.465831136209786],
        "probability_of_touch": [1.0, 1.0, 0.9661799112521838],
        "delta": [1.0, 0.6039490632362865, -0.525237550169406],
        "gamma": [0.0, 0.015297136732317718, 0.015806160944019643],
        "theta": [0.0, -0.21821351060901806, 0.22301627833773927],
        "vega": [0.0, 0.20095091693287098, 0.20763771616023433],
        "rho": [0.0, 0.08536880237502181, -0.07509774107468528],
    }


def test_calendar_spread():
    stock_price = 127.14
    half_range = round(stock_price * 0.5, 2)
    payload = {
        "stock_price": stock_price,
        "start_date": "2021-01-18",
        "target_date": "2021-01-29",
        "volatility": 0.427,
        "interest_rate": 0.0009,
        "min_stock": stock_price - half_range,
        "max_stock": stock_price + half_range,
        "strategy": [
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
        ],
    }

    assert run_validated_strategy(payload) == {
        "probability_of_profit": 0.6002074818796856,
        "expected_profit_if_profitable": 1380.68,
        "expected_loss_if_unprofitable": -693.04,
        "profit_ranges": [(118.85, 136.17)],
        "per_leg_cost": [4600.0, -5900.0],
        "strategy_cost": -1300.0,
        "minimum_return_in_the_domain": -1300.0,
        "maximum_return_in_the_domain": 3010.5363361936493,
        "implied_volatility": [0.47300000000000003, 0.419],
        "in_the_money_probability": [0.4895105709759477, 0.4805997906939539],
        "probability_of_touch": [1.0, 1.0],
        "delta": [-0.5216914758915705, 0.5273457614638198],
        "gamma": [0.03882722919950356, 0.02669940508461828],
        "theta": [0.22727438444823292, -0.15634971608107964],
        "vega": [0.09571294014902997, 0.1389462831961853],
        "rho": [-0.022202087247849632, 0.046016214466188525],
    }
