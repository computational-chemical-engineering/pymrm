---
description: "environment: pymrm and numpy importable inside the eval sandbox"
tags: [env]
max_turns: 6
timeout_seconds: 300
allowed_tools: [Read, Glob, Grep]
---

Run exactly this shell command and nothing else:

python -c "import pymrm, numpy, scipy; print('PYMRM_OK', pymrm.__file__)"

Then end your final message with the full output line of that command. If the
command fails, end with the line ENV_BROKEN followed by the error.
