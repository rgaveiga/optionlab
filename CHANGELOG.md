# CHANGELOG

## 1.2.1 (2024-06-03)

- Add 1 to the difference between the target date and the start date, to consider the target date as a trading day.

## 1.2.0 (2024-03-31)

- Add functions to run engine.

## 1.1.0 (2024-03-24)

- Refactor the engine's `run` method for readability.
- Accept dictionary of inputs to `StratgyEngine` init.

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
