from __future__ import division
from __future__ import print_function

import datetime as dt
from typing import Any

from numpy import array, ndarray, zeros

from optionlab.black_scholes import get_bs_info, get_implied_vol
from optionlab.models import (
    Inputs,
    Action,
    OptionStrategy,
    StockStrategy,
    ClosedPosition,
    Outputs,
    ProbabilityOfProfitInputs,
    ProbabilityOfProfitArrayInputs,
    OptionType,
    EngineData,
)
from optionlab.support import (
    get_pl_profile,
    get_pl_profile_stock,
    get_pl_profile_bs,
    get_profit_range,
    create_price_seq,
    create_price_samples,
    get_pop,
)
from optionlab.utils import get_nonbusiness_days, get_pl, pl_to_csv


def run_strategy(inputs_data: Inputs | dict) -> Outputs:
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
        stock_price_array=create_price_seq(inputs.min_stock, inputs.max_stock),
        terminal_stock_prices=array(inputs.array_prices or []),
        inputs=inputs,
    )

    data._days_in_year = 252 if inputs.discard_nonbusiness_days else 365

    if inputs.start_date and inputs.target_date:
        if inputs.discard_nonbusiness_days:
            n_discarded_days = get_nonbusiness_days(
                inputs.start_date, inputs.target_date, inputs.country
            )
        else:
            n_discarded_days = 0

        data.days_to_target = (
            inputs.target_date - inputs.start_date
        ).days - n_discarded_days
    else:
        data.days_to_target = inputs.days_to_target_date

    for i, strategy in enumerate(inputs.strategy):
        data.type.append(strategy.type)

        if isinstance(strategy, OptionStrategy):
            data.strike.append(strategy.strike)
            data.premium.append(strategy.premium)
            data.n.append(strategy.n)
            data.action.append(strategy.action)
            data._previous_position.append(strategy.prev_pos or 0.0)

            if not strategy.expiration:
                data._days_to_maturity.append(data.days_to_target)
                data._use_bs.append(False)
            elif isinstance(strategy.expiration, dt.date) and inputs.start_date:
                if inputs.discard_nonbusiness_days:
                    n_discarded_days = get_nonbusiness_days(
                        inputs.start_date, strategy.expiration, inputs.country
                    )
                else:
                    n_discarded_days = 0

                data._days_to_maturity.append(
                    (strategy.expiration - inputs.start_date).days - n_discarded_days
                )

                data._use_bs.append(strategy.expiration != inputs.target_date)
            elif isinstance(strategy.expiration, int):
                if strategy.expiration >= data.days_to_target:
                    data._days_to_maturity.append(strategy.expiration)

                    data._use_bs.append(strategy.expiration != data.days_to_target)
                else:
                    raise ValueError(
                        "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                    )
            else:
                raise ValueError("Expiration must be a date, an int or None.")

        elif isinstance(strategy, StockStrategy):
            data.n.append(strategy.n)
            data.action.append(strategy.action)
            data._previous_position.append(strategy.prev_pos or 0.0)
            data.strike.append(0.0)
            data.premium.append(0.0)
            data._use_bs.append(False)
            data._days_to_maturity.append(-1)

        elif isinstance(strategy, ClosedPosition):
            data._previous_position.append(strategy.prev_pos)
            data.strike.append(0.0)
            data.n.append(0)
            data.premium.append(0.0)
            data.action.append("n/a")
            data._use_bs.append(False)
            data._days_to_maturity.append(-1)
        else:
            raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

    return data


def _run(data: EngineData) -> EngineData:
    """
    run -> runs calculations for an options strategy.

    Returns
    -------
    output : Outputs
        An Outputs object containing the output of a calculation.
    """
    inputs = data.inputs

    time_to_target = (
        data.days_to_target + 1
    ) / data._days_in_year  # To consider the target date as a trading day
    data.cost = [0.0] * len(data.type)

    data.profit = zeros((len(data.type), data.stock_price_array.shape[0]))
    data.strategy_profit = zeros(data.stock_price_array.shape[0])

    if inputs.compute_expectation and data.terminal_stock_prices.shape[0] == 0:
        data.terminal_stock_prices = create_price_samples(
            inputs.stock_price,
            inputs.volatility,
            time_to_target,
            inputs.interest_rate,
            inputs.distribution,
            inputs.dividend_yield,
            inputs.mc_prices_number,
        )

    if data.terminal_stock_prices.shape[0] > 0:
        data.profit_mc = zeros((len(data.type), data.terminal_stock_prices.shape[0]))
        data.strategy_profit_mc = zeros(data.terminal_stock_prices.shape[0])

    for i, type in enumerate(data.type):
        if type in ("call", "put"):
            _run_option_calcs(data, i)
        elif type == "stock":
            _run_stock_calcs(data, i)
        elif type == "closed":
            _run_closed_position_calcs(data, i)

        data.strategy_profit += data.profit[i]

        if inputs.compute_expectation or inputs.distribution == "array":
            data.strategy_profit_mc += data.profit_mc[i]

    data._profit_ranges = get_profit_range(data.stock_price_array, data.strategy_profit)

    pop_inputs: ProbabilityOfProfitInputs | ProbabilityOfProfitArrayInputs
    if inputs.distribution in ("normal", "laplace", "black-scholes"):
        pop_inputs = ProbabilityOfProfitInputs(
            source=inputs.distribution,  # type: ignore
            stock_price=inputs.stock_price,
            volatility=inputs.volatility,
            years_to_maturity=time_to_target,
            interest_rate=inputs.interest_rate,
            dividend_yield=inputs.dividend_yield,
        )
    elif inputs.distribution == "array":
        pop_inputs = ProbabilityOfProfitArrayInputs(array=data.terminal_stock_prices)
    else:
        raise ValueError("Source not supported yet!")

    data.profit_probability = get_pop(data._profit_ranges, pop_inputs)

    if inputs.profit_target is not None:
        data._profit_target_range = get_profit_range(
            data.stock_price_array, data.strategy_profit, inputs.profit_target
        )
        data.profit_target_probability = get_pop(data._profit_target_range, pop_inputs)

    if inputs.loss_limit is not None:
        data._loss_limit_rangesm = get_profit_range(
            data.stock_price_array, data.strategy_profit, inputs.loss_limit + 0.01
        )
        data.loss_limit_probability = 1.0 - get_pop(data._loss_limit_ranges, pop_inputs)

    return data


def _run_option_calcs(data: EngineData, i: int) -> EngineData:
    inputs = data.inputs
    action: Action = data.action[i]  # type: ignore
    type: OptionType = data.type[i]  # type: ignore

    if data._previous_position[i] < 0.0:
        # Previous position is closed
        data.implied_volatility.append(0.0)
        data.itm_probability.append(0.0)
        data.delta.append(0.0)
        data.gamma.append(0.0)
        data.vega.append(0.0)
        data.theta.append(0.0)

        cost = (data.premium[i] + data._previous_position[i]) * data.n[i]

        if data.action[i] == "buy":
            cost *= -1.0

        data.cost[i] = cost
        data.profit[i] += cost

        if inputs.compute_expectation or inputs.distribution == "array":
            data.profit_mc[i] += cost

        return data

    time_to_maturity = (
        data._days_to_maturity[i] + 1
    ) / data._days_in_year  # To consider the expiration date as a trading day
    bs = get_bs_info(
        inputs.stock_price,
        data.strike[i],
        inputs.interest_rate,
        inputs.volatility,
        time_to_maturity,
        inputs.dividend_yield,
    )

    data.gamma.append(bs.gamma)
    data.vega.append(bs.vega)

    data.implied_volatility.append(
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

    negative_multiplier = 1 if data.action[i] == "buy" else -1

    if type == "call":
        data.itm_probability.append(bs.call_itm_prob)
        data.delta.append(bs.call_delta * negative_multiplier)
        data.theta.append(bs.call_theta / data._days_in_year * negative_multiplier)
    else:
        data.itm_probability.append(bs.put_itm_prob)
        data.delta.append(bs.put_delta * negative_multiplier)
        data.theta.append(bs.put_theta / data._days_in_year * negative_multiplier)

    if data._previous_position[i] > 0.0:  # Premium of the open position
        opt_value = data._previous_position[i]
    else:  # Current premium
        opt_value = data.premium[i]

    if data._use_bs[i]:
        target_to_maturity = (
            data._days_to_maturity[i] - data.days_to_target + 1
        ) / data._days_in_year  # To consider the expiration date as a trading day

        data.profit[i], data.cost[i] = get_pl_profile_bs(
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

        if inputs.compute_expectation or inputs.distribution == "array":
            data.profit_mc[i] = get_pl_profile_bs(
                type,
                action,
                data.strike[i],
                opt_value,
                inputs.interest_rate,
                target_to_maturity,
                inputs.interest_rate,
                data.n[i],
                data.terminal_stock_prices,
                inputs.dividend_yield,
                inputs.opt_commission,
            )[0]
    else:
        data.profit[i], data.cost[i] = get_pl_profile(
            type,
            action,
            data.strike[i],
            opt_value,
            data.n[i],
            data.stock_price_array,
            inputs.opt_commission,
        )

        if inputs.compute_expectation or inputs.distribution == "array":
            data.profit_mc[i] = get_pl_profile(
                type,
                action,
                data.strike[i],
                opt_value,
                data.n[i],
                data.terminal_stock_prices,
                inputs.opt_commission,
            )[0]

    return data


def _run_stock_calcs(data: EngineData, i: int) -> EngineData:
    inputs = data.inputs
    action: Action = data.action[i]  # type: ignore

    data.implied_volatility.append(0.0)
    data.itm_probability.append(1.0)
    data.delta.append(1.0)
    data.gamma.append(0.0)
    data.vega.append(0.0)
    data.theta.append(0.0)

    if data._previous_position[i] < 0.0:  # Previous position is closed
        costtmp = (inputs.stock_price + data._previous_position[i]) * data.n[i]

        if data.action[i] == "buy":
            costtmp *= -1.0

        data.cost[i] = costtmp
        data.profit[i] += costtmp

        if inputs.compute_expectation or inputs.distribution == "array":
            data.profit_mc[i] += costtmp

        return data

    if data._previous_position[i] > 0.0:  # Stock price at previous position
        stockpos = data._previous_position[i]
    else:  # Spot price of the stock at start date
        stockpos = inputs.stock_price

    data.profit[i], data.cost[i] = get_pl_profile_stock(
        stockpos,
        action,
        data.n[i],
        data.stock_price_array,
        inputs.stock_commission,
    )

    if inputs.compute_expectation or inputs.distribution == "array":
        data.profit_mc[i] = get_pl_profile_stock(
            stockpos,
            action,
            data.n[i],
            data.terminal_stock_prices,
            inputs.stock_commission,
        )[0]

    return data


def _run_closed_position_calcs(data: EngineData, i: int) -> EngineData:
    inputs = data.inputs

    data.implied_volatility.append(0.0)
    data.itm_probability.append(0.0)
    data.delta.append(0.0)
    data.gamma.append(0.0)
    data.vega.append(0.0)
    data.theta.append(0.0)

    data.cost[i] = data._previous_position[i]
    data.profit[i] += data._previous_position[i]

    if inputs.compute_expectation or inputs.distribution == "array":
        data.profit_mc[i] += data._previous_position[i]

    return data


def _generate_outputs(data: EngineData) -> Outputs:
    inputs = data.inputs
    optional_outputs: dict[str, Any] = {}

    if inputs.profit_target is not None:
        optional_outputs["probability_of_profit_target"] = (
            data.profit_target_probability
        )
        optional_outputs["profit_target_ranges"] = data._profit_target_range

    if inputs.loss_limit is not None:
        optional_outputs["probability_of_loss_limit"] = data.loss_limit_probability

    if (
        inputs.compute_expectation or inputs.distribution == "array"
    ) and data.terminal_stock_prices.shape[0] > 0:
        profit = data.strategy_profit_mc[data.strategy_profit_mc >= 0.01]
        loss = data.strategy_profit_mc[data.strategy_profit_mc < 0.0]
        optional_outputs["average_profit_from_mc"] = 0.0
        optional_outputs["average_loss_from_mc"] = (
            loss.mean() if loss.shape[0] > 0 else 0.0
        )

        if profit.shape[0] > 0:
            optional_outputs["average_profit_from_mc"] = profit.mean()

        if loss.shape[0] > 0:
            optional_outputs["average_loss_from_mc"] = loss.mean()

        optional_outputs["probability_of_profit_from_mc"] = (
            data.strategy_profit_mc >= 0.01
        ).sum() / data.strategy_profit_mc.shape[0]

    return Outputs.model_validate(
        optional_outputs
        | {
            "inputs": inputs,
            "data": data,
            "probability_of_profit": data.profit_probability,
            "strategy_cost": sum(data.cost),
            "per_leg_cost": data.cost,
            "profit_ranges": data._profit_ranges,
            "minimum_return_in_the_domain": data.strategy_profit.min(),
            "maximum_return_in_the_domain": data.strategy_profit.max(),
            "implied_volatility": data.implied_volatility,
            "in_the_money_probability": data.itm_probability,
            "delta": data.delta,
            "gamma": data.gamma,
            "theta": data.theta,
            "vega": data.vega,
        }
    )


class StrategyEngine:
    def __init__(self, inputs_data: Inputs | dict):
        """
        __init__ -> initializes class variables.

        Returns
        -------
        None.
        """
        inputs = (
            inputs_data
            if isinstance(inputs_data, Inputs)
            else Inputs.model_validate(inputs_data)
        )

        self.data = _init_inputs(inputs)

    def run(self) -> Outputs:
        """
        run -> runs calculations for an options strategy.

        Returns
        -------
        output : Outputs
            An Outputs object containing the output of a calculation.
        """

        self.data = _run(self.data)

        return _generate_outputs(self.data)

    def get_pl(self, leg: int | None = None) -> tuple[ndarray, ndarray]:
        """
        get_pl -> returns the profit/loss profile of either a leg or the whole
        strategy.

        Parameters
        ----------
        leg : int, optional
            Index of the leg. Default is None (whole strategy).

        Returns
        -------
        stock prices : numpy array
            Sequence of stock prices within the bounds of the stock price domain.
        P/L profile : numpy array
            Profit/loss profile of either a leg or the whole strategy.
        """

        return get_pl(self.data, leg)

    def pl_to_csv(self, filename: str = "pl.csv", leg: int | None = None) -> None:
        """
        pl_to_csv -> saves the profit/loss data to a .csv file.

        Parameters
        ----------
        filename : string, optional
            Name of the .csv file. Default is 'pl.csv'.
        leg : int, optional
            Index of the leg. Default is None (whole strategy).

        Returns
        -------
        None.
        """

        pl_to_csv(self.data, filename, leg)
