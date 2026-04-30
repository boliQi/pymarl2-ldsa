# Role Assignment Framework for SMAC

This module implements a dynamic role assignment mechanism using LLMs. Agents are assigned roles (e.g., Attacker, Defender) based on the current game state, and these roles are encoded into vectors to augment the agent's observations.

## Structure

- `role_config.py`: Defines available roles (`Attacker`, `Defender`, etc.) and their descriptions.
- `role_manager.py`: Main controller. Handles LLM queries and maintains current role state.
- `state_descriptor.py`: Converts game observations into natural language for the LLM.
- `llm_adapter.py`: Interface to the LLM service.

## Integration Guide

To use this framework in your PyMARL/HARL training loop:

### 1. Initialize RoleManager

In your runner (e.g., `pymarl/src/runners/episode_runner.py`) or controller:

```python
from role_assign.role_manager import RoleManager

# Inside your runner's __init__ or setup
self.role_manager = RoleManager(
    n_agents=self.env_info["n_agents"],
    n_enemies=self.env_info["n_enemies"],
    update_interval=50,  # Update roles every 50 steps
    api_key="your_key_here"
)
```

### 2. Update Roles During Episode

In your main episode loop (e.g., inside `run()`):

```python
# Inside the step loop
self.role_manager.update_roles(t, obs_list=current_obs, global_state=state)
```

### 3. Augment Observations

Before passing observations to your agent network, append the role encoding:

```python
# Get encodings for all agents
role_encodings = self.role_manager.get_all_encodings() # Shape: (n_agents, encoding_dim)

# Concatenate with original observations
# Assuming obs is (n_agents, obs_dim)
import numpy as np
augmented_obs = np.concatenate([obs, role_encodings], axis=1)

# Pass augmented_obs to your agent/mac
actions = self.mac.select_actions(augmented_obs, ...)
```

### 4. Update Network Input Dimension

Ensure your agent's network definition (e.g., in `pymarl/src/modules/agents/rnn_agent.py`) accepts the larger input size:

```python
# input_shape = original_obs_shape + role_encoding_dim
```

## Customization

- **Roles**: Edit `role_config.py` to add new roles or change descriptions.
- **State Description**: Modify `state_descriptor.py` to provide more detailed game context to the LLM.
