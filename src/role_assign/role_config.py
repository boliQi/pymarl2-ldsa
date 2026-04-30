from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

class RoleType(Enum):
    """
    Enum for defined agent roles specifically for SMAC 5m_vs_6m.
    对应: 抗伤、逃跑、输出、救人
    """
    SHIELD = "shield"       # 抗伤 (The Tank)
    SURVIVOR = "survivor"   # 逃跑 (The Wounded/Kiter)
    ANCHOR = "anchor"       # 输出 (The DPS)
    DEAD = "dead"           # 死亡 (Dead)

@dataclass
class RoleConfig:
    """Configuration for a specific role."""
    role_type: RoleType
    role_id: int
    name: str
    description: str
    behavior_guidelines: str
    
    def to_prompt_str(self) -> str:
        return f"""
Role Name: {self.name}
Description: {self.description}
Behavior Guidelines: {self.behavior_guidelines}
"""

# Predefined SMAC Roles (Optimized for Marine Micro)
SMAC_ROLES = {
    RoleType.SHIELD: RoleConfig(
        role_type=RoleType.SHIELD,
        role_id=0,
        name="Shield",
        description="High HP unit positioned to absorb enemy fire.",
        behavior_guidelines="Move slightly forward to attract aggro. Hold position or move laterally. Do not retreat unless HP drops critically."
    ),
    RoleType.SURVIVOR: RoleConfig(
        role_type=RoleType.SURVIVOR,
        role_id=1,
        name="Survivor",
        description="Critical HP unit that must avoid damage at all costs.",
        behavior_guidelines="Execute Hard Retreat. Move away from nearest enemies immediately. Only attack if safe (Kiting)."
    ),
    RoleType.ANCHOR: RoleConfig(
        role_type=RoleType.ANCHOR,
        role_id=2,
        name="Anchor",
        description="Safe unit focusing on maximizing damage output.",
        behavior_guidelines="Stand ground or minimize movement. Focus fire on the target with the lowest HP. Maintain concave formation."
    ),
    RoleType.DEAD: RoleConfig(
        role_type=RoleType.DEAD,
        role_id=3,
        name="Dead",
        description="Agent is dead (HP=0).",
        behavior_guidelines="No action. This role indicates the agent has been eliminated."
    )
}

class RoleEncoder:
    """Encodes roles into vector representations."""
    def __init__(self, encoding_dim: int = 4, use_one_hot: bool = True):
        self.encoding_dim = encoding_dim
        self.use_one_hot = use_one_hot
        self.num_roles = len(SMAC_ROLES)

    def encode(self, role_type: RoleType) -> np.ndarray:
        if role_type not in SMAC_ROLES:
            raise ValueError(f"Unknown RoleType: {role_type}")
            
        role_id = SMAC_ROLES[role_type].role_id
        
        if self.use_one_hot:
            encoding = np.zeros(self.num_roles, dtype=np.float32)
            encoding[role_id] = 1.0
            return encoding
        else:
            # Placeholder for embedding lookup
            vec = np.zeros(self.encoding_dim, dtype=np.float32)
            vec[role_id % self.encoding_dim] = 1.0 
            return vec