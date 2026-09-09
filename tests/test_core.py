import datetime as dt

import pytest

from optionlab import Inputs, run_strategy
from optionlab.black_scholes import get_bs_info
from optionlab.models import Outputs
from tests.profit_reference import assert_reference, reference


# Probabilities, ranges and conditional returns are checked against independent
# root solving and quadrature; snapshots retain the other outputs.
OUTPUT_EXCLUDE_FIELDS = {
    "expected_profit_if_profitable",
    "expected_loss_if_unprofitable",
    "data",
    "inputs",
    "probability_of_profit",
    "profit_ranges",
    "probability_of_profit_target",
    "profit_target_ranges",
    "probability_of_loss_limit",
    "loss_limit_ranges",
}


def assert_approx_equal(actual, expected, key=None):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()

        for key, value in expected.items():
            assert_approx_equal(actual[key], value, key)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(actual, expected):
            assert_approx_equal(actual_item, expected_item, key)
    else:
        if key == "implied_volatility":
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
            return

        assert actual == pytest.approx(expected)


COVERED_CALL_RESULT = {
    "per_leg_cost": [-16899.0, 409.99999999999994],
    "strategy_cost": -16489.0,
    "minimum_return_in_the_domain": -9590.000000000002,
    "maximum_return_in_the_domain": 2011.0,
    "implied_volatility": [0.0, 0.45621031663425116],
    "in_the_money_probability": [1.0, 0.256866624586934],
    "probability_of_touch": [1.0, 0.5277250352054264],
    "delta": [1.0, -0.30713817729665704],
    "gamma": [0.0, 0.013948977387090415],
    "theta": [0.0, 0.19283555235589467],
    "vega": [0.0, 0.1832408146218486],
    "rho": [0.0, -0.04506390742751745],
}

PROB_100_ITM_RESULT = {
    "per_leg_cost": [-750.0, 990.0],
    "strategy_cost": 240.0,
    "minimum_return_in_the_domain": 240.0,
    "maximum_return_in_the_domain": 740.0000000000018,
    "implied_volatility": [0.4942372292738584, 0.4826500896570693],
    "in_the_money_probability": [0.54558925139931, 0.465831136209786],
    "probability_of_touch": [1.0, 0.9661799112521838],
    "delta": [0.6039490632362865, -0.525237550169406],
    "gamma": [0.015297136732317718, 0.015806160944019643],
    "theta": [-0.21821351060901806, 0.22301627833773927],
    "vega": [0.20095091693287098, 0.20763771616023433],
    "rho": [0.08536880237502181, -0.07509774107468528],
}

NAKED_CALL = {
    "per_leg_cost": [114.99999999999999],
    "strategy_cost": 114.99999999999999,
    "minimum_return_in_the_domain": -6991.999999999999,
    "maximum_return_in_the_domain": 114.99999999999999,
    "implied_volatility": [0.2557726289266796],
    "in_the_money_probability": [0.1832371984432129],
    "probability_of_touch": [0.3741546603689868],
    "delta": [-0.20371918274704337],
    "gamma": [0.023104402361599465],
    "theta": [0.091289876347897],
    "vega": [0.12750177318341913],
    "rho": [-0.02417676577711979],
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


PREVIOUS_POSITION_CALL_BASE = {
    "stock_price": 168.99,
    "start_date": "2023-01-16",
    "target_date": "2023-02-17",
    "volatility": 0.483,
    "interest_rate": 0.045,
    "min_stock": 68.99,
    "max_stock": 268.99,
}


def run_validated_strategy(payload):
    outputs = run_strategy(Inputs.model_validate(payload))

    assert isinstance(outputs, Outputs)
    assert_reference(outputs)

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


def test_selected_calculations(nvidia):
    payload = nvidia | {
        "strategy": with_expiration(COVERED_CALL_LEGS, nvidia["target_date"]),
        "calculations": ["PoP"],
    }

    outputs = run_strategy(payload)

    assert outputs.probability_of_profit == pytest.approx(
        reference(outputs)[0][0], abs=1e-9, rel=0
    )
    assert outputs.expected_profit_if_profitable == pytest.approx(0.0)
    assert outputs.expected_loss_if_unprofitable == pytest.approx(0.0)
    assert outputs.implied_volatility == []
    assert outputs.delta == []

    payload["calculations"] = ["expectation", "impvol"]
    outputs = run_strategy(payload)

    assert outputs.probability_of_profit == pytest.approx(0.0)
    assert outputs.profit_ranges == []
    _, means, _ = reference(outputs)
    assert outputs.expected_profit_if_profitable == pytest.approx(
        means[0], abs=0.00051, rel=0
    )
    assert outputs.expected_loss_if_unprofitable == pytest.approx(
        means[1], abs=0.00051, rel=0
    )
    assert outputs.implied_volatility == pytest.approx(
        COVERED_CALL_RESULT["implied_volatility"], rel=1e-6, abs=1e-6
    )
    assert outputs.delta == []


def test_black_scholes():
    bs = get_bs_info(
        s=100.0,
        x=105.0,
        r=1.0 / 100,
        vol=20.0 / 100,
        years_to_maturity=60 / 365,
        y=0.0,
    )

    assert bs.call_price == pytest.approx(BLACK_SCHOLES_EXPECTED["call_price"])
    assert bs.call_delta == pytest.approx(BLACK_SCHOLES_EXPECTED["call_delta"])
    assert bs.call_theta == pytest.approx(BLACK_SCHOLES_EXPECTED["call_theta"])
    assert bs.call_rho == pytest.approx(BLACK_SCHOLES_EXPECTED["call_rho"])
    assert bs.call_itm_prob == pytest.approx(BLACK_SCHOLES_EXPECTED["call_itm_prob"])
    assert bs.call_prob_of_touch == pytest.approx(
        BLACK_SCHOLES_EXPECTED["call_prob_of_touch"]
    )
    assert bs.put_price == pytest.approx(BLACK_SCHOLES_EXPECTED["put_price"])
    assert bs.put_delta == pytest.approx(BLACK_SCHOLES_EXPECTED["put_delta"])
    assert bs.put_theta == pytest.approx(BLACK_SCHOLES_EXPECTED["put_theta"])
    assert bs.put_rho == pytest.approx(BLACK_SCHOLES_EXPECTED["put_rho"])
    assert bs.put_itm_prob == pytest.approx(BLACK_SCHOLES_EXPECTED["put_itm_prob"])
    assert bs.put_prob_of_touch == pytest.approx(
        BLACK_SCHOLES_EXPECTED["put_prob_of_touch"]
    )
    assert bs.gamma == pytest.approx(BLACK_SCHOLES_EXPECTED["gamma"])
    assert bs.vega == pytest.approx(BLACK_SCHOLES_EXPECTED["vega"])


def test_covered_call(nvidia):
    payload = nvidia | {
        "strategy": with_expiration(COVERED_CALL_LEGS, nvidia["target_date"])
    }

    assert_approx_equal(run_validated_strategy(payload), COVERED_CALL_RESULT)


def test_covered_call_w_days_to_target(nvidia):
    payload = nvidia | {
        "start_date": None,
        "target_date": None,
        "days_to_target_date": 24,  # 32 days minus 9 non-business days plus 1 to consider the expiration date
        "strategy": COVERED_CALL_LEGS,
    }

    assert_approx_equal(run_validated_strategy(payload), COVERED_CALL_RESULT)


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

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [-15899.0, 409.99999999999994],
            "strategy_cost": -15489.0,
            "minimum_return_in_the_domain": -8590.000000000002,
            "maximum_return_in_the_domain": 3011.0,
            "implied_volatility": [0.0, 0.45621031663425116],
            "in_the_money_probability": [1.0, 0.256866624586934],
            "probability_of_touch": [1.0, 0.5277250352054264],
            "delta": [1.0, -0.30713817729665704],
            "gamma": [0.0, 0.013948977387090415],
            "theta": [0.0, 0.19283555235589467],
            "vega": [0.0, 0.1832408146218486],
            "rho": [0.0, -0.04506390742751745],
        },
    )


@pytest.mark.parametrize("explicit_expiration", [False, True])
def test_nonsimultaneous_call_spread(nvidia, explicit_expiration):
    """The notebook's prior call basis produces a credit and profit at all prices."""
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

    if not explicit_expiration:
        for leg in payload["strategy"]:
            leg.pop("expiration")

    assert_approx_equal(run_validated_strategy(payload), PROB_100_ITM_RESULT)


@pytest.mark.parametrize(
    "half_range, minimum_return",
    [(100.0, -8091.0), (round(168.99 * 0.5, 2), -6541.0)],
    ids=["wide-domain", "notebook"],
)
def test_short_straddle(half_range, minimum_return):
    """Cover the notebook's price domain as well as the wider regression domain."""
    inputs = Inputs(
        stock_price=168.99,
        volatility=0.483,
        start_date=dt.date(2023, 1, 16),
        target_date=dt.date(2023, 2, 17),
        interest_rate=0.045,
        min_stock=168.99 - half_range,
        max_stock=168.99 + half_range,
        strategy=[
            {
                "type": "call",
                "strike": 170.0,
                "premium": 9.9,
                "n": 100,
                "action": "sell",
            },
            {
                "type": "put",
                "strike": 170.0,
                "premium": 10.2,
                "n": 100,
                "action": "sell",
            },
        ],
    )

    assert_approx_equal(
        run_validated_strategy(inputs),
        {
            "per_leg_cost": [990.0, 1019.9999999999999],
            "strategy_cost": 2010.0,
            "minimum_return_in_the_domain": minimum_return,
            "maximum_return_in_the_domain": 2010.0,
            "implied_volatility": [0.4826500896570693, 0.48346942266415105],
            "in_the_money_probability": [0.465831136209786, 0.534168863790214],
            "probability_of_touch": [0.9661799112521838, 1.0],
            "delta": [-0.525237550169406, 0.474762449830594],
            "gamma": [0.015806160944019643, 0.015806160944019643],
            "theta": [0.22301627833773927, 0.19278895912917046],
            "vega": [0.20763771616023433, 0.20763771616023433],
            "rho": [-0.07509774107468528, 0.0861146280376816],
        },
    )


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

    assert_approx_equal(run_validated_strategy(payload), NAKED_CALL)


def test_previous_naked_call():
    payload = PREVIOUS_POSITION_CALL_BASE | {
        "strategy": [
            {
                "type": "call",
                "strike": 175.0,
                "premium": 7.55,
                "n": 100,
                "action": "buy",
                "prev_pos": 9.00,
            }
        ]
    }

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [-900.0],
            "strategy_cost": -900.0,
            "minimum_return_in_the_domain": -900.0,
            "maximum_return_in_the_domain": 8499.0,
            "implied_volatility": [0.47185720287639016],
            "in_the_money_probability": [0.3896519029956125],
            "probability_of_touch": [0.8052162544137671],
            "delta": [0.4478206614305777],
            "gamma": [0.01570219883634658],
            "theta": [-0.21968576560850298],
            "vega": [0.2062720173874019],
            "rho": [0.06466425659962557],
        },
    )


def test_previous_call_combined_with_currently_sold_call():
    payload = PREVIOUS_POSITION_CALL_BASE | {
        "strategy": [
            {
                "type": "call",
                "strike": 175.0,
                "premium": 7.55,
                "n": 100,
                "action": "buy",
                "prev_pos": 9.00,
            },
            {
                "type": "call",
                "strike": 180.0,
                "premium": 5.65,
                "n": 100,
                "action": "sell",
            },
        ]
    }

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [-900.0, 565.0],
            "strategy_cost": -335.0,
            "minimum_return_in_the_domain": -335.0,
            "maximum_return_in_the_domain": 165.00000000000182,
            "implied_volatility": [0.47185720287639016, 0.4643544255095575],
            "in_the_money_probability": [0.3896519029956125, 0.31945606955896366],
            "probability_of_touch": [0.8052162544137671, 0.6580872103707619],
            "delta": [0.4478206614305777, -0.37442226491472497],
            "gamma": [0.01570219883634658, 0.015046586930711309],
            "theta": [-0.21968576560850298, 0.20911925871992107],
            "vega": [0.2062720173874019, 0.19765956814968416],
            "rho": [0.06466425659962557, -0.05452969743627417],
        },
    )


def test_sold_call_considering_previous_loss():
    payload = PREVIOUS_POSITION_CALL_BASE | {
        "strategy": [
            {"type": "closed", "prev_pos": -50.0},
            {
                "type": "call",
                "strike": 180.0,
                "premium": 5.65,
                "n": 100,
                "action": "sell",
            },
        ]
    }

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [-50.0, 565.0],
            "strategy_cost": 515.0,
            "minimum_return_in_the_domain": -8384.0,
            "maximum_return_in_the_domain": 515.0,
            "implied_volatility": [0.0, 0.4643544255095575],
            "in_the_money_probability": [0.0, 0.31945606955896366],
            "probability_of_touch": [0.0, 0.6580872103707619],
            "delta": [0.0, -0.37442226491472497],
            "gamma": [0.0, 0.015046586930711309],
            "theta": [0.0, 0.20911925871992107],
            "vega": [0.0, 0.19765956814968416],
            "rho": [0.0, -0.05452969743627417],
        },
    )


def test_bought_call_using_previous_profit():
    payload = PREVIOUS_POSITION_CALL_BASE | {
        "strategy": [
            {"type": "closed", "prev_pos": 565.0},
            {
                "type": "call",
                "strike": 180.0,
                "premium": 5.65,
                "n": 100,
                "action": "buy",
            },
        ]
    }

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [565.0, -565.0],
            "strategy_cost": 0.0,
            "minimum_return_in_the_domain": 0.0,
            "maximum_return_in_the_domain": 8899.0,
            "implied_volatility": [0.0, 0.4643544255095575],
            "in_the_money_probability": [0.0, 0.31945606955896366],
            "probability_of_touch": [0.0, 0.6580872103707619],
            "delta": [0.0, 0.37442226491472497],
            "gamma": [0.0, 0.015046586930711309],
            "theta": [0.0, -0.20911925871992107],
            "vega": [0.0, 0.19765956814968416],
            "rho": [0.0, 0.05452969743627417],
        },
    )


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

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [-15899.0, -750.0, 990.0],
            "strategy_cost": -15659.0,
            "minimum_return_in_the_domain": -8760.000000000002,
            "maximum_return_in_the_domain": 11740.0,
            "implied_volatility": [0.0, 0.4942372292738584, 0.4826500896570693],
            "in_the_money_probability": [1.0, 0.54558925139931, 0.465831136209786],
            "probability_of_touch": [1.0, 1.0, 0.9661799112521838],
            "delta": [1.0, 0.6039490632362865, -0.525237550169406],
            "gamma": [0.0, 0.015297136732317718, 0.015806160944019643],
            "theta": [0.0, -0.21821351060901806, 0.22301627833773927],
            "vega": [0.0, 0.20095091693287098, 0.20763771616023433],
            "rho": [0.0, 0.08536880237502181, -0.07509774107468528],
        },
    )


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

    assert_approx_equal(
        run_validated_strategy(payload),
        {
            "per_leg_cost": [4600.0, -5900.0],
            "strategy_cost": -1300.0,
            "minimum_return_in_the_domain": -1300.0,
            "maximum_return_in_the_domain": 3010.5363361936493,
            "implied_volatility": [0.4727644953287636, 0.4187446787999526],
            "in_the_money_probability": [0.4895105709759477, 0.4805997906939539],
            "probability_of_touch": [1.0, 1.0],
            "delta": [-0.5216914758915705, 0.5273457614638198],
            "gamma": [0.03882722919950356, 0.02669940508461828],
            "theta": [0.22727438444823292, -0.15634971608107964],
            "vega": [0.09571294014902997, 0.1389462831961853],
            "rho": [-0.022202087247849632, 0.046016214466188525],
        },
    )
