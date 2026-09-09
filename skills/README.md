# OptionLab agent skills

[optionlab-strategy](optionlab-strategy/SKILL.md) teaches coding agents to construct
`run_strategy` inputs, interpret `Outputs`, and plot P/L. It is written in English
to match the repository documentation and instructs agents to answer in the user's
language. It includes all input fields, leg types, output fields and runnable examples.

[optionlab-black-scholes](optionlab-black-scholes/SKILL.md) covers standalone call and put prices and delta, gamma, theta, vega, and rho, plus probability of touch and implied volatility, with calculator code, units, strike arrays, solver limits, and boundary cases.

## Install

To install the calculator skill, replace `optionlab-strategy` with `optionlab-black-scholes` in the paths and commands below. Copy the entire chosen skill directory, including any references.

Copy the entire `optionlab-strategy` directory, including `references/`, to one
of the following destinations. Paths are relative to the consuming project for
project installs; `~` denotes the user's home for personal installs.

| Agent | Project destination | Personal destination |
| --- | --- | --- |
| Codex | `.agents/skills/optionlab-strategy/` | `~/.agents/skills/optionlab-strategy/` |
| Claude Code | `.claude/skills/optionlab-strategy/` | `~/.claude/skills/optionlab-strategy/` |
| OpenCode | `.opencode/skills/optionlab-strategy/` | `~/.config/opencode/skills/optionlab-strategy/` |

These discovery locations are documented by [Codex](https://developers.openai.com/codex/skills/),
[Claude Code](https://code.claude.com/docs/en/skills), and
[OpenCode](https://opencode.ai/docs/skills/). OpenCode also recognizes the `.agents`
and `.claude` skill locations. Use one destination per agent to avoid duplicate
copies. Other agents supporting the `SKILL.md` format can use the same folder in
their documented skill directory. No agent-specific tools or plugins are required.

For example, from this repository's root, install for Claude Code in this project
with PowerShell (choose a destination from the table for other agents):

```powershell
New-Item -ItemType Directory -Force -Path .claude/skills | Out-Null
Copy-Item -LiteralPath skills/optionlab-strategy -Destination .claude/skills/optionlab-strategy -Recurse
```

Or with a POSIX shell:

```sh
mkdir -p .claude/skills
cp -R skills/optionlab-strategy .claude/skills/optionlab-strategy
```

These examples assume the destination skill does not exist yet. For an update,
replace its contents deliberately instead of nesting a second copy. The source
folder `skills/` is a distribution directory, not automatic agent configuration.
Start a new agent session after installation if it does not discover the skill.
Installing instructions does not install the Python library: use the project's
OptionLab environment, or install `optionlab` in the consuming Python environment.
