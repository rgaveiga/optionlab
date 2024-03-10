from optionlab.strategy import Strategy


def test_covered_call(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-2-mixing-options-and-stocks-1e9f59f388f

    inputs = nvidia | {
        # The covered call strategy is defined
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

    st = Strategy()
    st.getdata(**inputs)
    outputs = st.run()

    assert outputs == {
        "ProbabilityOfProfit": 0.5489826392738772,
        "StrategyCost": -16489.0,
        "PerLegCost": [-16899.0, 409.99999999999994],
        "ProfitRanges": [[164.9, float("inf")]],
        "MinimumReturnInTheDomain": -9590.000000000002,
        "MaximumReturnInTheDomain": 2011.0,
    }


def test_covered_call_w_prev_position(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-2-mixing-options-and-stocks-1e9f59f388f

    inputs = nvidia | {
        # The covered call strategy is defined
        "strategy": [
            {"type": "stock", "n": 100, "action": "buy", "prevpos": 158.99},
            {
                "type": "call",
                "strike": 185.0,
                "premium": 4.1,
                "n": 100,
                "action": "sell",
            },
        ]
    }

    st = Strategy()
    st.getdata(**inputs)
    outputs = st.run()

    assert outputs == {
        "ProbabilityOfProfit": 0.7094641281976972,
        "StrategyCost": -15489.0,
        "PerLegCost": [-15899.0, 409.99999999999994],
        "ProfitRanges": [[154.9, float("inf")]],
        "MinimumReturnInTheDomain": -8590.000000000002,
        "MaximumReturnInTheDomain": 3011.0,
    }


def test_100_perc_itm(nvidia):
    # https://medium.com/@rgaveiga/python-for-options-trading-3-a-trade-with-100-probability-of-profit-886e934addbf

    inputs = nvidia | {
        # The covered call strategy is defined
        "strategy": [
            {
                "type": "call",
                "strike": 165.0,
                "premium": 12.65,
                "n": 100,
                "action": "buy",
                "prevpos": 7.5,
                "expiration": nvidia["targetdate"],
            },
            {
                "type": "call",
                "strike": 170.0,
                "premium": 9.9,
                "n": 100,
                "action": "sell",
                "expiration": nvidia["targetdate"],
            },
        ]
    }

    st = Strategy()
    st.getdata(**inputs)
    outputs = st.run()

    # Print useful information on screen
    assert outputs == {
        "ProbabilityOfProfit": 1.0,
        "StrategyCost": 240.0,
        "PerLegCost": [-750.0, 990.0],
        "ProfitRanges": [[0.0, float("inf")]],
        "MinimumReturnInTheDomain": 240.0,
        "MaximumReturnInTheDomain": 740.0000000000018,
    }
