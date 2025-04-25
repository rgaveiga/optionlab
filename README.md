![OptionLab](optionlab.png)

# OptionLab

This package is a lightweight library written entirely in Python, designed to provide 
quick evaluation of option strategy ideas.

The code produces various outputs, including the profit/loss profile of the strategy on 
a user-defined target date, the range of stock prices for which the strategy is 
profitable (i.e., generating a return greater than \$0.01), the Greeks associated with 
each leg of the strategy using the Black-Sholes model, the resulting debit or credit on the 
trading account, the maximum and minimum returns within a specified lower and higher price 
range of the underlying asset, and an estimate of the strategy's probability of profit.

If you have any questions, corrections, comments or suggestions, just 
[drop a message](mailto:roberto.veiga@ufabc.edu.br).

You can also reach me on [Linkedin](https://www.linkedin.com/in/roberto-gomes-phd-8a718317b/) or 
follow me on [X](https://x.com/rgaveiga). When I have some free time, which is rare, I publish articles 
on [Medium](https://medium.com/@rgaveiga).

If you want to support this and other open source projects that I maintain, become a 
[sponsor on Github](https://github.com/sponsors/rgaveiga).

## Installation

The easiest way to install **OptionLab** is using **pip**:

```
pip install optionlab
```

## Documentation

You can access the API documentation for **OptionLab** on the [project's GitHub Pages site](https://rgaveiga.github.io/optionlab).

## Contributions

Contributions are definitely welcome. However, it should be mentioned that this 
repository uses [poetry](https://python-poetry.org/) as a package manager and 
[git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) with 
[pre-commit](https://pre-commit.com/) to customize actions on the repository. Source 
code must be formatted using [black](https://github.com/psf/black).

## Disclaimer

This is free software and is provided as is. The author makes no guarantee that its 
results are accurate and is not responsible for any losses caused by the use of the 
code.

Options are very risky derivatives and, like any other type of financial vehicle, 
trading options requires due diligence. This code is provided for educational and 
research purposes only.

Bugs can be reported as issues.
