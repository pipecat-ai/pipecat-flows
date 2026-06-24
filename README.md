<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/pipecat-flows.png">
</div></h1>

# Pipecat Flows is now part of Pipecat

As of **pipecat-ai 1.5.0**, Pipecat Flows ships inside the core [`pipecat-ai`](https://github.com/pipecat-ai/pipecat) package under the `pipecat.flows` namespace. There is no longer a separate package to install or keep version-matched.

This `pipecat-ai-flows` package is **deprecated** and frozen at its final release — it will not receive further updates. New features and fixes land in `pipecat.flows`.

## Migrate

Install Pipecat (Flows is included — nothing extra to add) and update your imports:

```python
# Before
from pipecat_flows import ContextStrategyConfig, FlowManager, NodeConfig
from pipecat_flows.types import ActionConfig, ContextStrategy

# After
from pipecat.flows import ContextStrategyConfig, FlowManager, NodeConfig
from pipecat.flows.types import ActionConfig, ContextStrategy
```

That's the only change — the API is the same.

## Where things moved

- **Framework** → [`pipecat/flows`](https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/flows)
- **Examples** → [`examples/flows`](https://github.com/pipecat-ai/pipecat/tree/main/examples/flows)
- **Documentation** → [Pipecat Flows guide](https://docs.pipecat.ai/guides/features/pipecat-flows)

## History

The release history of the standalone package remains here in [CHANGELOG.md](./CHANGELOG.md). Going forward, changes are tracked in Pipecat's [changelog](https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md).

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Pipecat Flows guide](https://docs.pipecat.ai/guides/features/pipecat-flows)

➡️ [Reach us on X](https://x.com/pipecat_ai)
