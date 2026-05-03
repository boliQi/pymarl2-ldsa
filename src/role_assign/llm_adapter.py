import os
import sys

# Ensure project root is importable so `language.call_llm` can be resolved.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from language.call_llm import LLM
except ImportError:
    LLM = None


class LLMAdapter:
    def __init__(
        self,
        api_key: str = None,
        use_big_model: bool = False,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = None,
    ):
        self.api_key = api_key
        self.use_big_model = use_big_model
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens

        if LLM is None:
            raise ImportError(
                "Failed to import 'language.call_llm.LLM'. "
                "Please ensure 'language/call_llm.py' exists under the project root "
                "and PYTHONPATH includes the project root."
            )

        self.llm = LLM(mode='openai')

    def query(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        return self.llm.call_llm(
            prompt,
            big_model=self.use_big_model,
            temperature=0.0,
            enable_thinking=self.enable_thinking,
            thinking_budget_tokens=self.thinking_budget_tokens,
        )

    def get_role_assignment(self, state_description: str, available_roles: list, num_agents: int, current_roles: list, map_name: str = "5m_vs_6m") -> list:
        """
        Constructs a prompt and parses the response to get a list of roles.

        Args:
            state_description: current state description
            available_roles: available role configs
            num_agents: number of agents
            current_roles: current role names for each agent
            map_name: current map name
        """
        if current_roles is None:
            print("No current roles provided, initializing to default 'Shield' roles.")

        roles_str = ", ".join([r.name for r in available_roles])

        prompt_5m_vs_6m = f"""
# Environment: StarCraft Multi-Agent Challenge (SMAC) - Map: {map_name}
# Scenario Context: {map_name} (Homogeneous Marines)
You are controlling 5 Marines against 6 enemy Marines.
CRITICAL TACTIC: You CANNOT win by brute force. You must use **Health Rotation**.
- **The Goal**: Distribute damage across all allies. Ideally, all 5 allies survive with low HP, rather than 1 ally dying early.
- **The Mechanism**: High HP units must step forward to shield Low HP units. Low HP units must retreat to the backline but keep shooting.
"""
        prompt_3s5z = f"""
# Environment: StarCraft Multi-Agent Challenge (SMAC) - Map: {map_name}
# Scenario Context: {map_name} (Heterogeneous: 3 Stalkers, 5 Zealots)
You are controlling 3 Stalkers (ranged) and 5 Zealots (melee) against enemies.
Protoss units have **Shields** in addition to Health. Shields regenerate out of combat.
CRITICAL TACTIC: Protect the Stalkers and use Zealots as a frontline shield.
- **The Goal**: Keep the fragile high-DPS Stalkers alive by kiting, while Zealots absorb damage and hold the frontline.
- **The Mechanism**: Zealots with high Health/Shield must act as Shields. Low Health Zealots should retreat slightly. Stalkers should generally be Anchors (DPS) unless heavily targeted, in which case they become Survivors and kite.
"""
        prompt_mmm2 = f"""
# Environment: StarCraft Multi-Agent Challenge (SMAC) - Map: {map_name}
# Scenario Context: {map_name} (Heterogeneous: Marines, Marauders, Medivacs)
You are controlling a mixed Terran bio force against enemies.
CRITICAL TACTIC: Maximize Medivac healing efficiency and focus fire with Marauders.
- **The Goal**: Protect the Medivacs, use Marines as a meat shield or DPS, and use Marauders to tank and deal heavy damage.
- **The Mechanism**: Units being targeted and dropping in HP should retreat (Survivor) to get healed by Medivacs. Healthy frontline units act as Shields. Unthreatened units act as Anchors.
"""
        prompt_fallback = f"""
# Environment: StarCraft Multi-Agent Challenge (SMAC) - Map: {map_name}
# Scenario Context: {map_name} (Generic Combat)
You are controlling a team of units in a StarCraft II battle.
CRITICAL TACTIC: Minimize casualties by rotating frontline units.
- **The Goal**: Distribute damage and focus fire.
- **The Mechanism**: Units with high combined Health/Shield tank (Shield). Low HP targeted units retreat (Survivor). Free units deal steady damage (Anchor).
"""

        if map_name == "5m_vs_6m":
            scenario_prompt = prompt_5m_vs_6m
        elif map_name == "3s5z":
            scenario_prompt = prompt_3s5z
        elif map_name == "MMM2":
            scenario_prompt = prompt_mmm2
        else:
            scenario_prompt = prompt_fallback

        prompt = f"""{scenario_prompt}
# Role Decision Logic (HP-driven)
Use [HP Analysis] in Current Game State as the primary decision source.

1. **Dead**:
    - If an agent is dead or its HP is 0, role must be Dead.

2. **Global Gate**:
    - If low_hp_exists=False, do NOT assign SHIELD or SURVIVOR.
    - In this case, every alive agent should be ANCHOR.

3. **SURVIVOR (Wounded)**:
    - Only choose from "Survivor candidates".
    - These are selected from the two lowest-HP allies with enough relative HP gap.
    - Prefer the lowest HP first.

4. **SHIELD (Protector)**:
    - Only choose from "Shield candidates".
    - These have sufficient HP advantage over the weakest ally and are near vulnerable allies.

5. **ANCHOR (Default DPS)**:
    - Any alive agent that is not Dead / Survivor / Shield should be ANCHOR.

Important:
- Prioritize team HP balancing by rotating damage away from lowest-HP allies.
- Keep the number of SURVIVOR and SHIELD minimal when HP differences are small.

Current Game State:
{state_description}

Available Roles:
{roles_str}

Task:
Assign a role to each of the {num_agents} agents to maximize team chance of winning.
If an agent is dead (HP=0), you MUST assign the role "Dead".
The returned role list MUST match agent id order exactly: index 0 is Agent 0, index 1 is Agent 1, etc.

Output Format:
Return a comma-separated list of roles, one for each agent (e.g., "Shield, Survivor, Dead, Shield, Dead").
Do not include any other text.
"""
        response = self.query(prompt)

        cleaned_response = response.strip().replace("\n", "").replace(".", "").replace("[", "").replace("]", "")
        tokens = cleaned_response.split(",")

        final_roles = []
        for token in tokens:
            cleaned_token = token.strip().strip("'").strip('"')
            if ":" in cleaned_token:
                cleaned_token = cleaned_token.split(":")[-1].strip()
            final_roles.append(cleaned_token)

        if len(final_roles) != num_agents:
            print(f"LLM returned {len(final_roles)} roles, expected {num_agents}. Using default.")
            print(f"LLM returned {final_roles}")
            return current_roles

        return final_roles
