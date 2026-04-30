from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import flags
#
# FLAGS = flags.FLAGS
# FLAGS(["main.py"])
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env
from .wrapper import StarCraftCapabilityEnvWrapper

__all__ = ["MultiAgentEnv", "StarCraft2Env", "StarCraftCapabilityEnvWrapper"]
