import sys
import os

# === 新增: 添加父目录到 sys.path 以便找到 language 包 ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ======================================================

# Try to import the existing LLM agent
# Assuming the project root is in PYTHONPATH
try:
    from language.call_llm import LLM
except ImportError:
    # Fallback or mock if running in isolation
    print("Warning: Could not import 'language.call_llm'. Using MockLLMAgent.")
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
        # 实际项目中这里可以替换为真实的 OpenAI 客户端
        if LLM:
            # Assuming gpt_agent constructor signature based on context
            # You might need to adjust arguments based on actual call_llm.py
            self.llm = LLM(mode='openai')
        else:
            self.llm = None

    def query(self, prompt: str) -> str:
        if self.llm:
            # Using the 'ask' method from the user's existing code
            # === [修改] 将之前 mock 的调用替换为真实调用 ===
            response = self.llm.call_llm(
                prompt,
                big_model=self.use_big_model,
                temperature=0.0,
                enable_thinking=self.enable_thinking,
                thinking_budget_tokens=self.thinking_budget_tokens,
            )
            return response
            # ======================================
        else:
            # === 修改: 返回 3 个角色以匹配测试用例 ===
            print("query with no llm")
            return "Attacker, Defender, Supporter"
            # ======================================

    def get_role_assignment(self, state_description: str, available_roles: list, num_agents: int, current_roles: list, map_name: str = "5m_vs_6m") -> list:
        """
        Constructs a prompt and parses the response to get a list of roles.
      
        Args:
            state_description: 当前状态描述
            available_roles: 可用角色列表
            num_agents: 智能体数量
            current_roles: 当前各智能体的角色名称列表（用于保持）
            map_name: 当前地图名称
        """
        if current_roles is None:
            print(f"!!!!!!!!!No current roles provided, initializing to default 'Shield' roles.")
        
        roles_str = ", ".join([r.name for r in available_roles])
        
        # --- Prompt Definitions ---
        
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
        
        # Simple parsing logic
        cleaned_response = response.strip().replace("\n", "").replace(".", "").replace("[", "").replace("]", "")
        tokens = cleaned_response.split(",")
        
        # 二次清洗：去除每个 token 两端的空白和可能的引号
        final_roles = []
        for token in tokens:
            cleaned_token = token.strip().strip("'").strip('"')
            if ":" in cleaned_token:
                cleaned_token = cleaned_token.split(":")[-1].strip()
            final_roles.append(cleaned_token)
        
        # Fallback if parsing fails
        if len(final_roles) != num_agents:
            print(f"LLM returned {len(final_roles)} roles, expected {num_agents}. Using default.")
            print(f"LLM returned {final_roles}")
            return current_roles
            
        return final_roles
