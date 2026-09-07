"""
This module implements the `run_strategy` function.

Given input data provided as either an `optionlab.models.Inputs` object or a dictionary,
`run_strategy` returns the results of an options strategy calculation (e.g., the
probability of profit on the target date) as an `optionlab.models.Outputs` object.
"""

from __future__ import division
from __future__ import print_function

import datetime as dt
from functools import partial

import numpy as np
from numpy import full, ndarray, zeros, array


from optionlab.black_scholes import get_bs_info, get_implied_vol
from optionlab.models import (
    Inputs,
    Action,
    Option,
    Stock,
    ClosedPosition,
    Outputs,
    BlackScholesModelInputs,
    ArrayInputs,
    OptionType,
    EngineData,
    PoPOutputs,
)
from optionlab.support import (
    get_pl_profile,
    get_pl_profile_stock,
    get_pl_profile_bs,
    create_price_seq,
    get_pop,
)
from optionlab.utils import get_nonbusiness_days
from optionlab.profit import array_segments, profile_pop, adaptive_segments


def _has_calculation(inputs: Inputs, calculation: str) -> bool:
    calculations = {"pop" if item == "PoP" else item for item in inputs.calculations}
    return calculation in calculations


def run_strategy(inputs_data: Inputs | dict) -> Outputs:
    """
    Runs the calculation for a strategy.

    ### Parameters

    `inputs_data`: input data used in the strategy calculation.

    ### Returns

    Output data from the strategy calculation.
    """

    inputs = (
        inputs_data
        if isinstance(inputs_data, Inputs)
        else Inputs.model_validate(inputs_data)
    )

    data = _init_inputs(inputs)

    data = _run(data)

    return _generate_outputs(data)


def _init_inputs(inputs: Inputs) -> EngineData:
    data = EngineData(
        stock_price_array=create_price_seq(
            inputs.min_stock, inputs.max_stock, inputs.price_step
        ),
        terminal_stock_prices=inputs.array if inputs.model == "array" else array([]),
        inputs=inputs,
    )

    data.days_in_year = (
        inputs.business_days_in_year if inputs.discard_nonbusiness_days else 365
    )

    if inputs.start_date and inputs.target_date:
        if inputs.discard_nonbusiness_days:
            n_discarded_days = get_nonbusiness_days(
                inputs.start_date, inputs.target_date, inputs.country
            )
        else:
            n_discarded_days = 0

        data.days_to_target = (
            (inputs.target_date - inputs.start_date).days + 1 - n_discarded_days
        )
    else:
        data.days_to_target = inputs.days_to_target_date

    for i, strategy in enumerate(inputs.strategy):
        data.type.append(strategy.type)

        if isinstance(strategy, Option):
            data.strike.append(strategy.strike)
            data.premium.append(strategy.premium)
            data.n.append(strategy.n)
            data.action.append(strategy.action)
            data.previous_position.append(strategy.prev_pos or 0.0)

            if not strategy.expiration:
                data.days_to_maturity.append(data.days_to_target)
                data.use_bs.append(False)
            elif isinstance(strategy.expiration, dt.date) and inputs.start_date:
                if inputs.discard_nonbusiness_days:
                    n_discarded_days = get_nonbusiness_days(
                        inputs.start_date, strategy.expiration, inputs.country
                    )
                else:
                    n_discarded_days = 0

                data.days_to_maturity.append(
                    (strategy.expiration - inputs.start_date).days
                    + 1
                    - n_discarded_days
                )

                data.use_bs.append(strategy.expiration != inputs.target_date)
            elif isinstance(strategy.expiration, int):
                if strategy.expiration >= data.days_to_target:
                    data.days_to_maturity.append(strategy.expiration)

                    data.use_bs.append(strategy.expiration != data.days_to_target)
                else:
                    raise ValueError(
                        "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                    )
            else:
                raise ValueError("Expiration must be a date, an int or None.")

        elif isinstance(strategy, Stock):
            data.n.append(strategy.n)
            data.action.append(strategy.action)
            data.previous_position.append(strategy.prev_pos or 0.0)
            data.strike.append(0.0)
            data.premium.append(0.0)
            data.use_bs.append(False)
            data.days_to_maturity.append(-1)

        elif isinstance(strategy, ClosedPosition):
            data.previous_position.append(strategy.prev_pos)
            data.strike.append(0.0)
            data.n.append(0)
            data.premium.append(0.0)
            data.action.append("n/a")
            data.use_bs.append(False)
            data.days_to_maturity.append(-1)
        else:
            raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

    return data


def _run(data: EngineData) -> EngineData:
    inputs = data.inputs

    time_to_target = data.days_to_target / data.days_in_year
    data.cost = [0.0] * len(data.type)

    data.strategy_profit = zeros(data.stock_price_array.shape[0])
    data.profit = np.empty(
        (len(data.type), data.stock_price_array.shape[0]), dtype=float
    )

    mc_leg_profit: ndarray | None = None
    if inputs.model == "array":
        data.strategy_profit_mc = zeros(data.terminal_stock_prices.shape[0])
        mc_leg_profit = np.empty_like(data.strategy_profit_mc)

    calculate_pop = _has_calculation(inputs, "pop")
    calculate_expectation = _has_calculation(inputs, "expectation")
    pop_inputs: BlackScholesModelInputs | ArrayInputs
    pop_out: PoPOutputs

    for i, type in enumerate(data.type):
        if type in ("call", "put"):
            leg_profit = _run_option_calcs(data, i, mc_leg_profit)
        elif type == "stock":
            leg_profit = _run_stock_calcs(data, i, mc_leg_profit)
        elif type == "closed":
            leg_profit = _run_closed_position_calcs(data, i)

        data.profit[i] = leg_profit
        np.add(data.strategy_profit, leg_profit, out=data.strategy_profit)

    if calculate_pop or calculate_expectation:
        if inputs.model == "black-scholes":
            pop_inputs = BlackScholesModelInputs(
                stock_price=inputs.stock_price,
                volatility=inputs.volatility,
                years_to_target_date=time_to_target,
                interest_rate=inputs.interest_rate,
                dividend_yield=inputs.dividend_yield,
            )
        elif inputs.model == "array":
            pop_inputs = ArrayInputs(array=data.strategy_profit_mc)
        else:
            raise ValueError("Model is not valid!")

        pop_calculator = (
            partial(_strategy_pop, data) if inputs.model == "black-scholes" else get_pop
        )
        pop_out = pop_calculator(
            data.stock_price_array,
            data.strategy_profit,
            pop_inputs,
            calculate_expectation=calculate_expectation,
        )

        if calculate_pop:
            data.profit_probability = pop_out.probability_of_reaching_target
            data.profit_ranges = pop_out.reaching_target_range

        if calculate_expectation:
            data.expected_profit = pop_out.expected_return_above_target
            data.expected_loss = pop_out.expected_return_below_target

        if (
            calculate_pop
            and inputs.profit_target is not None
            and inputs.profit_target > 0.01
        ):
            pop_out_prof_targ = pop_calculator(
                data.stock_price_array,
                data.strategy_profit,
                pop_inputs,
                inputs.profit_target,
                calculate_expectation=False,
            )
            data.profit_target_probability = (
                pop_out_prof_targ.probability_of_reaching_target
            )
            data.profit_target_ranges = pop_out_prof_targ.reaching_target_range

        if calculate_pop and inputs.loss_limit is not None and inputs.loss_limit < 0.0:
            pop_out_loss_lim = pop_calculator(
                data.stock_price_array,
                data.strategy_profit,
                pop_inputs,
                inputs.loss_limit + 0.01,
                calculate_expectation=False,
            )
            data.loss_limit_probability = pop_out_loss_lim.probability_of_missing_target
            data.loss_limit_ranges = pop_out_loss_lim.missing_target_range

    return data


def _strategy_pop(data, s, profit, inputs, target=0.01, calculate_expectation=True):
    """Evaluate the same legs independently of the public plotting domain."""
    scratch = data.model_copy(deep=False)
    scratch.inputs = data.inputs.model_copy(
        update={"calculations": [], "model": "black-scholes"}
    )
    scratch.cost = list(data.cost)

    def evaluate(prices):
        scratch.stock_price_array = prices
        total = np.zeros_like(prices)
        for i, kind in enumerate(data.type):
            if kind in ("call", "put"):
                total += _run_option_calcs(scratch, i)
            elif kind == "stock":
                total += _run_stock_calcs(scratch, i)
            else:
                total += _run_closed_position_calcs(scratch, i)
        return total

    knots = []
    live = {}
    lower_slope = upper_slope = 0.0
    upper_intercept = float(evaluate(np.array([0.0]))[0])
    lower_intercept = upper_intercept
    for i, kind in enumerate(data.type):
        if data.previous_position[i] < 0 or kind == "closed":
            continue
        n = data.n[i] * (1 if data.action[i] == "buy" else -1)
        if kind == "stock":
            lower_slope += n
            upper_slope += n
            continue
        tau = max(
            0.0, (data.days_to_maturity[i] - data.days_to_target) / data.days_in_year
        )
        discount = np.exp(-inputs.dividend_yield * tau)
        strike_value = data.strike[i] * np.exp(-inputs.interest_rate * tau)
        knots.append(strike_value / discount)
        if kind == "put":
            lower_slope -= n * discount
            upper_intercept -= n * strike_value
        else:
            upper_slope += n * discount
            upper_intercept -= n * strike_value
        if tau > 0 and inputs.volatility > 0:
            # Calls and puts have the same curvature; net identical exposures.
            key = (data.strike[i], tau)
            live[key] = live.get(key, 0) + n
    live = {key: n for key, n in live.items() if n != 0}

    if not live:
        nodes = np.unique([0.0, *knots, max([inputs.stock_price, *knots]) * 2])
        segments = array_segments(nodes, evaluate(nodes))
    else:

        def bounds(nodes, interior):
            error = np.zeros(len(nodes) + 1)
            for (strike, tau), n in live.items():
                w = inputs.volatility * np.sqrt(tau)
                drift = (
                    inputs.interest_rate
                    - inputs.dividend_yield
                    + inputs.volatility**2 / 2
                ) * tau
                # Gamma is unimodal in log price; clamp its maximizer to each cell.
                mode = strike * np.exp(-drift - w * w)
                point = np.clip(mode, nodes[:-1], nodes[1:])
                d1 = (np.log(point / strike) + drift) / w
                gamma = np.exp(-inputs.dividend_yield * tau - d1 * d1 / 2) / (
                    point * w * np.sqrt(2 * np.pi)
                )
                error[1:-1] += abs(n) * gamma * np.diff(nodes) ** 2 / 8
                # OTM prices bound deviation from the exact asymptotes over
                # each entire tail, so no unbounded payoff moment is omitted.
                for index, kind, price in (
                    (0, "call", nodes[0]),
                    (-1, "put", nodes[-1]),
                ):
                    residual = get_pl_profile_bs(
                        kind,
                        "buy",
                        strike,
                        0.0,
                        inputs.interest_rate,
                        tau,
                        inputs.volatility,
                        1,
                        np.array([price]),
                        inputs.dividend_yield,
                    )[0][0]
                    error[index] += abs(n) * abs(residual)
            tails = [
                [0.0, nodes[0], lower_slope, lower_intercept],
                [nodes[-1], np.inf, upper_slope, upper_intercept],
            ]
            return np.concatenate(([tails[0]], interior, [tails[1]])), error

        segments = adaptive_segments(
            evaluate, bounds, knots, inputs, target, calculate_expectation
        )
    return profile_pop(segments, inputs, target, calculate_expectation, evaluate)


def _run_option_calcs(
    data: EngineData, i: int, mc_out: ndarray | None = None
) -> ndarray:
    inputs = data.inputs
    action: Action = data.action[i]  # type: ignore
    type: OptionType = data.type[i]  # type: ignore
    calculate_impvol = _has_calculation(inputs, "impvol")
    calculate_greeks = _has_calculation(inputs, "greeks")

    if data.previous_position[i] < 0.0:
        # Previous position is closed
        if calculate_impvol:
            data.implied_volatility.append(0.0)

        if calculate_greeks:
            data.itm_probability.append(0.0)
            data.probability_of_touch.append(0.0)
            data.delta.append(0.0)
            data.gamma.append(0.0)
            data.vega.append(0.0)
            data.theta.append(0.0)
            data.rho.append(0.0)

        cost = (data.premium[i] + data.previous_position[i]) * data.n[i]

        if data.action[i] == "buy":
            cost *= -1.0

        data.cost[i] = cost

        if inputs.model == "array":
            data.strategy_profit_mc += cost

        return full(data.stock_price_array.shape[0], cost)

    time_to_maturity = data.days_to_maturity[i] / data.days_in_year

    if calculate_impvol:
        data.implied_volatility.append(
            float(
                get_implied_vol(
                    type,
                    data.premium[i],
                    inputs.stock_price,
                    data.strike[i],
                    inputs.interest_rate,
                    time_to_maturity,
                    inputs.dividend_yield,
                )
            )
        )

    if calculate_greeks:
        bs = get_bs_info(
            inputs.stock_price,
            data.strike[i],
            inputs.interest_rate,
            inputs.volatility,
            time_to_maturity,
            inputs.dividend_yield,
        )

        data.gamma.append(
            float(bs.gamma)
        )  # TODO: This is required because of mypy. Check later for workarounds, maybe using zero-dimensional numpy arrays
        data.vega.append(float(bs.vega))

        negative_multiplier = 1 if data.action[i] == "buy" else -1

        if type == "call":
            data.itm_probability.append(float(bs.call_itm_prob))
            data.probability_of_touch.append(float(bs.call_prob_of_touch))
            data.delta.append(float(bs.call_delta * negative_multiplier))
            data.theta.append(
                float(bs.call_theta / data.days_in_year * negative_multiplier)
            )
            data.rho.append(float(bs.call_rho * negative_multiplier))
        else:
            data.itm_probability.append(float(bs.put_itm_prob))
            data.probability_of_touch.append(float(bs.put_prob_of_touch))
            data.delta.append(float(bs.put_delta * negative_multiplier))
            data.theta.append(
                float(bs.put_theta / data.days_in_year * negative_multiplier)
            )
            data.rho.append(float(bs.put_rho * negative_multiplier))

    if data.previous_position[i] > 0.0:  # Premium of the open position
        opt_value = data.previous_position[i]
    else:  # Current premium
        opt_value = data.premium[i]

    if data.use_bs[i]:
        target_to_maturity = (
            data.days_to_maturity[i] - data.days_to_target
        ) / data.days_in_year  # To consider the expiration date as a trading day

        leg_profit, data.cost[i] = get_pl_profile_bs(
            type,
            action,
            data.strike[i],
            opt_value,
            inputs.interest_rate,
            target_to_maturity,
            inputs.volatility,
            data.n[i],
            data.stock_price_array,
            inputs.dividend_yield,
            inputs.opt_commission,
        )

        if inputs.model == "array":
            assert mc_out is not None
            mc_profile = get_pl_profile_bs(
                type,
                action,
                data.strike[i],
                opt_value,
                inputs.interest_rate,
                target_to_maturity,
                inputs.volatility,
                data.n[i],
                data.terminal_stock_prices,
                inputs.dividend_yield,
                inputs.opt_commission,
                out=mc_out,
            )[0]
            np.add(data.strategy_profit_mc, mc_profile, out=data.strategy_profit_mc)
    else:
        leg_profit, data.cost[i] = get_pl_profile(
            type,
            action,
            data.strike[i],
            opt_value,
            data.n[i],
            data.stock_price_array,
            inputs.opt_commission,
        )

        if inputs.model == "array":
            assert mc_out is not None
            mc_profile = get_pl_profile(
                type,
                action,
                data.strike[i],
                opt_value,
                data.n[i],
                data.terminal_stock_prices,
                inputs.opt_commission,
                out=mc_out,
            )[0]
            np.add(data.strategy_profit_mc, mc_profile, out=data.strategy_profit_mc)

    return leg_profit  # type: ignore


def _run_stock_calcs(
    data: EngineData, i: int, mc_out: ndarray | None = None
) -> ndarray:
    inputs = data.inputs
    action: Action = data.action[i]  # type: ignore
    calculate_impvol = _has_calculation(inputs, "impvol")
    calculate_greeks = _has_calculation(inputs, "greeks")

    if calculate_greeks:
        if action == "buy":
            data.delta.append(1.0)
        else:
            data.delta.append(-1.0)

        data.itm_probability.append(1.0)
        data.probability_of_touch.append(1.0)
        data.gamma.append(0.0)
        data.vega.append(0.0)
        data.rho.append(0.0)
        data.theta.append(0.0)

    if calculate_impvol:
        data.implied_volatility.append(0.0)

    if data.previous_position[i] < 0.0:  # Previous position is closed
        costtmp = (inputs.stock_price + data.previous_position[i]) * data.n[i]

        if data.action[i] == "buy":
            costtmp *= -1.0

        data.cost[i] = costtmp

        if inputs.model == "array":
            data.strategy_profit_mc += costtmp

        return full(data.stock_price_array.shape[0], costtmp)

    if data.previous_position[i] > 0.0:  # Stock price at previous position
        stockpos = data.previous_position[i]
    else:  # Spot price of the stock at start date
        stockpos = inputs.stock_price

    leg_profit, data.cost[i] = get_pl_profile_stock(
        stockpos,
        action,
        data.n[i],
        data.stock_price_array,
        inputs.stock_commission,
    )

    if inputs.model == "array":
        assert mc_out is not None
        mc_profile = get_pl_profile_stock(
            stockpos,
            action,
            data.n[i],
            data.terminal_stock_prices,
            inputs.stock_commission,
            out=mc_out,
        )[0]
        np.add(data.strategy_profit_mc, mc_profile, out=data.strategy_profit_mc)

    return leg_profit


def _run_closed_position_calcs(data: EngineData, i: int) -> ndarray:
    inputs = data.inputs

    if _has_calculation(inputs, "impvol"):
        data.implied_volatility.append(0.0)

    if _has_calculation(inputs, "greeks"):
        data.itm_probability.append(0.0)
        data.probability_of_touch.append(0.0)
        data.delta.append(0.0)
        data.gamma.append(0.0)
        data.vega.append(0.0)
        data.rho.append(0.0)
        data.theta.append(0.0)

    data.cost[i] = data.previous_position[i]

    if inputs.model == "array":
        data.strategy_profit_mc += data.previous_position[i]

    return full(data.stock_price_array.shape[0], data.previous_position[i])


def _generate_outputs(data: EngineData) -> Outputs:
    return Outputs(
        inputs=data.inputs,
        data=data,
        probability_of_profit=data.profit_probability,
        expected_profit_if_profitable=data.expected_profit,
        expected_loss_if_unprofitable=data.expected_loss,
        strategy_cost=sum(data.cost),
        per_leg_cost=data.cost,
        profit_ranges=data.profit_ranges,
        minimum_return_in_the_domain=data.strategy_profit.min(),
        maximum_return_in_the_domain=data.strategy_profit.max(),
        implied_volatility=data.implied_volatility,
        in_the_money_probability=data.itm_probability,
        probability_of_touch=data.probability_of_touch,
        delta=data.delta,
        gamma=data.gamma,
        theta=data.theta,
        vega=data.vega,
        rho=data.rho,
        probability_of_profit_target=data.profit_target_probability,
        probability_of_loss_limit=data.loss_limit_probability,
        profit_target_ranges=data.profit_target_ranges,
        loss_limit_ranges=data.loss_limit_ranges,
    )
