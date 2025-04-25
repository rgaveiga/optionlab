# CHANGELOG

## 1.4.3 (2025-04-14)

- Updated docstrings.
- Added documentation with `pdoc`.
- Changed __init__.py for compatibility with `pdoc` autodocumentation.
- Removed `BaseLeg` from models.py.
- Changed `StrategyType` to `StrategyLegType` in models.py for clarity.
- Removed "normal" as an alias for "black-scholes" to avoid confusion with Bachelier model.
- Updated Readme.md.

## 1.4.2 (2025-01-25)

- Removed `expected_profit` and `expected_loss` calculation from `_get_pop_bs` in support.py; implementation was not correct, giving wrong results when compared with Monte Carlo simulations

## 1.4.1 (2025-01-04)

- Removed a small bug in `create_price_seq` in support.py
- Improved the algorithm in `get_profit_range` in support.py, then renamed to `_get_profit_range`
- Created a helper function `_get_sign_changes` in support.py, called by `get_profit_range`
- Removed the fields `probability_of_profit_from_mc`, `average_profit_from_mc` and `average_loss_from_mc` from `Outputs` in models.py
- Created the fields `expected_profit` and `expected_loss` in `Outputs` in models.py
- Created a class `PoPOutputs` in models.py containing fields returned by `get_pop` in support.py
- Removed Laplace form `get_pop` in support.py
- Improved `get_pop` in support.py to return a `PoPOutputs` object with more information
- Added naked calls as an example of strategy
- Created a custom type `FloatOrNdarray` that can contain a float or a numpy.ndarray in models.py
- Created the helper functions `_get_pop_bs` and `get_pop_array` in support.py

## 1.4.0 (2025-01-01)

- Changed the class name `DistributionInputs` to `TheoreticalModelInputs` in models.py, to be more descriptive
- Changed the class name `DistributionBlackScholesInputs` to `BlackScholesModelInputs` in models.py
- Changed the class name `DistributionLaplaceInputs` to `LaplaceInputs` in models.py
- Changed the class name `DistributionArrayInputs` to `ArrayInputs` in models.py
- Changed literal `Distribution` to `TheoreticalModel`
- Moved `create_price_samples` from support.py to a new module price_array.py and renamed it to `create_price_array`
- Commented a code snippet in engine.py where terminal stock prices are created using `create_price_samples`, to be removed in a next version
- Allowed a dictionary as input for `create_price_array` in price_array.py
- Allowed a dictionary as input for `get_pop` in support.py

## 1.3.5 (2024-12-28)

- Created a base class `DistributionInputs`
- Changed the name of `ProbabilityOfProfitInputs` in models.py (and everywhere in the code) to `DistributionBlackScholesInputs`, which inherits from `DistributionInputs`
- Removed the `source` field from `DistributionBlackScholesInputs`
- Modified interest_rate: float = Field(0.0, ge=0.0) in `DistributionBlackScholesInputs` in models.py
- Modified volatility: float = Field(gt=0.0) in `DistributionInputs` in models.py
- Modified years_to_maturity: float = Field(ge=0.0) in `DistributionInputs` in models.py
- Created a class `DistributionLaplaceInputs` in models.py, which inherits from `DistributionInputs`
- Changed `years_to_maturity` field in `DistributionInputs` to `years_to_target_date`
- Refactored `create_price_samples` in support.py
- Added __hash__ = object.__hash__ in `DistributionBlackScholesInputs` and `DistributionLaplaceInputs` in models.py to allow their use in `create_price_samples` in support.py with caching
- Updated tests to reflect those changes
- Removed a deprecated class, `StrategyEngine`, commented in a previous version
- Added a test for Laplace distribution
- Added a test for Calendar Spread

## 1.3.4 (2024-12-20)

- Deleted `OptionInfo` class in models.py, because it is not necessary
- Deleted `return_in_the_domain_ratio` in `Outputs` in models.py
- Deleted `Country` in models.py, because it is not necessary
- Deleted source: Literal["array"] = "array" in `ProbabilityOfProfitArrayInputs` class in models.py, because it is not necessary
- Strike prices in black_scholes.py functions now can be provided also as numpy arrays and those functions return numpy arrays
- `BlackScholesInfo` fields in models.py now can be both float and numpy arrays
- Split `get_d1_d2` function in black_scholes.py into two functions, `get_d1` and `get_d2`
- Added the field `business_days_in_year` in `Inputs` class in models.py to allow market-dependent customization; also changed in engine.py
- Added Greek Rho calculation to black-scholes.py
- Added `call_rho` and `put_rho` fields to `BlackScholesInfo` in models.py
- Added `rho` field to `EngineData` in models.py
- Added `rho` field to `Outputs` in models.py
- Added `rho` data field in engine.py
- Added a `seed` argument to `create_price_samples` in support.py to make the generation of price samples deterministic
- Changed `array_prices` field to simply `array` in `Inputs` in models.py
- Changed and commented some tests in test_core.py

## 1.3.3 (2024-12-18)

- Updated docstrings to comply with reStructuredText (RST) standards
- Changed the `country` argument in `get_nonbusiness_days` in utils.py to accept a string
- Changed the `data` argument in `get_pl` and `pl_to_csv` in utils.py to accept an `Outputs` object instead of `EngineData`
- Commented 'source: Literal["array"] = "array"' in `ProbabilityOfProfitArrayInputs` class in models.py, because `source` is not necessary
- Commented `OptionInfo` class in models.py, because it is not used anywhere
- Commented `return_in_the_domain_ratio` in `Outputs` in models.py, because it is not necessary
- Commented `Country` in models.py, because it is not necessary
- Changed country: Country = "US" to country: str = "US" in models.py

## 1.3.2 (2024-11-30)

- Changed Laplace distribution implementation in `create_price_samples` and `get_pop` functions in support.py

## 1.3.1 (2024-09-27)

- discriminator="type" removed from strategy: list[StrategyLeg] = Field(..., min_length=1) in models.py, since
it was causing errors in new Pydantic versions.
- Changed `StotckStrategy` and `OptionStrategy` to `Stock` and `Option` in models.py, respectively.
- Changed `BaseStrategy` to `BaseLeg` in models.py
- Changed `Strategy` to `StrategyLeg` in models.py
- Removed `premium` field from `Stock` in models.py
- Moved `n` field to `BaseLeg` in models.py

## 1.3.0 (2024-09-13)

- Remove the deprecated `StrategyEngine` class (it remains commented in the code).
- Update the README.md file to reflect the current state of the library

## 1.2.1 (2024-06-03)

- Add 1 to `time_to_target` and `time_to_maturity` in `engine.py` to consider the target and expiration dates as  trading days in the calculations
- Change Jupyter notebooks in the `examples` directory to utilize the `run_strategy()` function for performing options strategy calculations, instead of using the `StrategyEngine` class (deprecated) 
- Correct the PoP Calculator notebook
- Change the name of variable `project_target_ranges` in `models.py` and `engine.py` to `profit_target_ranges`

## 1.2.0 (2024-03-31)

- Add functions to run engine

## 1.1.0 (2024-03-24)

- Refactor the engine's `run` method for readability
- Accept dictionary of inputs to `StratgyEngine` init

## 1.0.1 (2024-03-18)

- Refactor __holidays__.py to a utils function using the `holiday` library

## 1.0.0 (2024-03-11)

**BREAKING CHANGES**:
- Renamed strategy.py to engine.py and `Strategy` to `StrategyEngine`
- Using pydantic for input validation into `StrategyEngine`
- Outputs are now also a Pydantic model
- Delete `use_dates`, as Pydantic will handle either using dates or `days_to_target`
- Renamed functions to be PEP8 compliant, i.e. instead of `getPoP`, now is `get_pop`
- Deleted options_chain.py module

## 0.1.7 (2023-07-04)

- Initial commit with strategy engine and examples
