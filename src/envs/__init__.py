from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

try:
    smac_v2 = True
    from .smac_v2 import StarCraftCapabilityEnvWrapper
except Exception as e:
    smac_v2 = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if smac_v2:
    REGISTRY["sc2v2"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
