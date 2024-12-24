# CHANGELOG

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
