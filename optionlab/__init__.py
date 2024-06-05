import typing


VERSION = "1.2.1"


if typing.TYPE_CHECKING:
    # import of virtually everything is supported via `__getattr__` below,
    # but we need them here for type checking and IDE support
    from .models import (
        Inputs,
        OptionType,
        OptionInfo,
        OptionStrategy,
        Outputs,
        ClosedPosition,
        ProbabilityOfProfitArrayInputs,
        ProbabilityOfProfitInputs,
        BlackScholesInfo,
        Distribution,
        Strategy,
        StrategyType,
        StockStrategy,
        Country,
        Action,
    )
    from .black_scholes import (
        get_itm_probability,
        get_implied_vol,
        get_option_price,
        get_d1_d2,
        get_bs_info,
        get_vega,
        get_delta,
        get_gamma,
        get_theta,
    )
    from .engine import StrategyEngine, run_strategy
    from .plot import plot_pl
    from .support import (
        get_pl_profile,
        get_pl_profile_stock,
        get_pl_profile_bs,
        create_price_seq,
        create_price_samples,
        get_profit_range,
        get_pop,
    )

__version__ = VERSION
__all__ = (
    # models
    "Inputs",
    "OptionType",
    "OptionInfo",
    "OptionStrategy",
    "Outputs",
    "ClosedPosition",
    "ProbabilityOfProfitArrayInputs",
    "ProbabilityOfProfitInputs",
    "BlackScholesInfo",
    "Distribution",
    "Strategy",
    "StrategyType",
    "StockStrategy",
    "Country",
    "Action",
    # engine
    "run_strategy",
    "StrategyEngine",
    # support
    "get_pl_profile",
    "get_pl_profile_stock",
    "get_pl_profile_bs",
    "create_price_seq",
    "create_price_samples",
    "get_profit_range",
    "get_pop",
    # black_scholes
    "get_d1_d2",
    "get_option_price",
    "get_itm_probability",
    "get_implied_vol",
    "get_bs_info",
    "get_vega",
    "get_delta",
    "get_gamma",
    "get_theta",
    # plot
    "plot_pl",
)

# A mapping of {<member name>: (package, <module name>)} defining dynamic imports
_dynamic_imports: "dict[str, tuple[str, str]]" = {
    # models
    "Inputs": (__package__, ".models"),
    "Outputs": (__package__, ".models"),
    "OptionType": (__package__, ".models"),
    "OptionInfo": (__package__, ".models"),
    "OptionStrategy": (__package__, ".models"),
    "ClosedPosition": (__package__, ".models"),
    "ProbabilityOfProfitArrayInputs": (__package__, ".models"),
    "ProbabilityOfProfitInputs": (__package__, ".models"),
    "BlackScholesInfo": (__package__, ".models"),
    "Distribution": (__package__, ".models"),
    "Strategy": (__package__, ".models"),
    "StrategyType": (__package__, ".models"),
    "StockStrategy": (__package__, ".models"),
    "Country": (__package__, ".models"),
    "Action": (__package__, ".models"),
    # engine
    "StrategyEngine": (__package__, ".engine"),
    "run_strategy": (__package__, ".engine"),
    # support
    "get_pl_profile": (__package__, ".support"),
    "get_pl_profile_stock": (__package__, ".support"),
    "get_pl_profile_bs": (__package__, ".support"),
    "create_price_seq": (__package__, ".support"),
    "create_price_samples": (__package__, ".support"),
    "get_profit_range": (__package__, ".support"),
    "get_pop": (__package__, ".support"),
    # black_scholes
    "get_d1_d2": (__package__, ".black_scholes"),
    "get_option_price": (__package__, ".black_scholes"),
    "get_itm_probability": (__package__, ".black_scholes"),
    "get_implied_vol": (__package__, ".black_scholes"),
    "get_bs_info": (__package__, ".black_scholes"),
    "get_vega": (__package__, ".black_scholes"),
    "get_delta": (__package__, ".black_scholes"),
    "get_gamma": (__package__, ".black_scholes"),
    "get_theta": (__package__, ".black_scholes"),
    # plot
    "plot_pl": (__package__, ".plot"),
}


def __getattr__(attr_name: str) -> object:
    dynamic_attr = _dynamic_imports[attr_name]

    package, module_name = dynamic_attr

    from importlib import import_module

    if module_name == "__module__":
        return import_module(f".{attr_name}", package=package)
    else:
        module = import_module(module_name, package=package)
        return getattr(module, attr_name)


def __dir__() -> "list[str]":
    return list(__all__)
