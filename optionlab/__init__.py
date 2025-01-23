import typing


VERSION = "1.4.1"


if typing.TYPE_CHECKING:
    # import of virtually everything is supported via `__getattr__` below,
    # but we need them here for type checking and IDE support
    from .models import (
        Inputs,
        OptionType,
        Option,
        Outputs,
        ClosedPosition,
        ArrayInputs,
        TheoreticalModelInputs,
        BlackScholesModelInputs,
        LaplaceInputs,
        BlackScholesInfo,
        TheoreticalModel,
        FloatOrNdarray,
        StrategyLeg,
        StrategyType,
        Stock,
        Action,
    )
    from .black_scholes import (
        get_itm_probability,
        get_implied_vol,
        get_option_price,
        get_d1,
        get_d2,
        get_bs_info,
        get_vega,
        get_delta,
        get_gamma,
        get_theta,
        get_rho,
    )
    from .engine import run_strategy
    from .plot import plot_pl
    from .price_array import create_price_array
    from .support import (
        get_pl_profile,
        get_pl_profile_stock,
        get_pl_profile_bs,
        create_price_seq,
        get_pop,
    )
    from .utils import (
        get_nonbusiness_days,
        get_pl,
        pl_to_csv,
    )

__version__ = VERSION
__all__ = (
    # models
    "Inputs",
    "OptionType",
    "Option",
    "Outputs",
    "ClosedPosition",
    "ArrayInputs",
    "TheoreticalModelInputs",
    "BlackScholesModelInputs",
    "LaplaceInputs",
    "BlackScholesInfo",
    "TheoreticalModel",
    "FloatOrNdarray",
    "StrategyLeg",
    "StrategyType",
    "Stock",
    "Action",
    # engine
    "run_strategy",
    # support
    "get_pl_profile",
    "get_pl_profile_stock",
    "get_pl_profile_bs",
    "create_price_seq",
    "get_pop",
    # black_scholes
    "get_d1",
    "get_d2",
    "get_option_price",
    "get_itm_probability",
    "get_implied_vol",
    "get_bs_info",
    "get_vega",
    "get_delta",
    "get_gamma",
    "get_theta",
    "get_rho",
    # plot
    "plot_pl",
    # price_array
    "create_price_array",
    # utils
    "get_nonbusiness_days",
    "get_pl",
    "pl_to_csv",
)

# A mapping of {<member name>: (package, <module name>)} defining dynamic imports
_dynamic_imports: "dict[str, tuple[str, str]]" = {
    # models
    "Inputs": (__package__, ".models"),
    "Outputs": (__package__, ".models"),
    "OptionType": (__package__, ".models"),
    "Option": (__package__, ".models"),
    "ClosedPosition": (__package__, ".models"),
    "ArrayInputs": (__package__, ".models"),
    "TheoreticalModelInputs": (__package__, ".models"),
    "BlackScholesModelInputs": (__package__, ".models"),
    "LaplaceInputs": (__package__, ".models"),
    "BlackScholesInfo": (__package__, ".models"),
    "TheoreticalModel": (__package__, ".models"),
    "FloatOrNdarray": (__package__, ".models"),
    "StrategyLeg": (__package__, ".models"),
    "StrategyType": (__package__, ".models"),
    "Stock": (__package__, ".models"),
    "Action": (__package__, ".models"),
    # engine
    "run_strategy": (__package__, ".engine"),
    # support
    "get_pl_profile": (__package__, ".support"),
    "get_pl_profile_stock": (__package__, ".support"),
    "get_pl_profile_bs": (__package__, ".support"),
    "create_price_seq": (__package__, ".support"),
    "get_pop": (__package__, ".support"),
    # black_scholes
    "get_d1": (__package__, ".black_scholes"),
    "get_d2": (__package__, ".black_scholes"),
    "get_option_price": (__package__, ".black_scholes"),
    "get_itm_probability": (__package__, ".black_scholes"),
    "get_implied_vol": (__package__, ".black_scholes"),
    "get_bs_info": (__package__, ".black_scholes"),
    "get_vega": (__package__, ".black_scholes"),
    "get_delta": (__package__, ".black_scholes"),
    "get_gamma": (__package__, ".black_scholes"),
    "get_theta": (__package__, ".black_scholes"),
    "get_rho": (__package__, ".black_scholes"),
    # plot
    "plot_pl": (__package__, ".plot"),
    # price_array
    "create_price_array": (__package__, ".price_array"),
    # utils
    "get_nonbusiness_days": (__package__, ".utils"),
    "get_pl": (__package__, ".utils"),
    "pl_to_csv": (__package__, ".utils"),
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
