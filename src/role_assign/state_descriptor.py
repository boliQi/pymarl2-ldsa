import numpy as np

class StateDescriptor:
    """
    Converts raw SMAC observations/states into natural language descriptions.
    """
    def __init__(
        self,
        n_agents,
        n_enemies,
        map_name="5m_vs_6m",
        low_hp_threshold=0.70,
        survivor_gap_threshold=0.20,
        shield_gap_threshold=0.20,
        protect_distance_threshold=0.1,
    ):
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.map_name = map_name
        self.low_hp_threshold = low_hp_threshold
        self.survivor_gap_threshold = survivor_gap_threshold
        self.shield_gap_threshold = shield_gap_threshold
        self.protect_distance_threshold = protect_distance_threshold
        
        # === 动态计算特征跨度 ===
        # PyMARL Global State 分配规则:
        # 盟友 (Ally): health, cooldown, x, y, [shield], [unit_type_1, ..., unit_type_n]
        # 敌军 (Enemy): health, x, y, [shield], [unit_type_1, ..., unit_type_n]
        
        self.shield_bits_ally = 0
        self.shield_bits_enemy = 0
        self.unit_type_bits = 0
        
        # 配置特定地图的额外特征位
        if map_name in ["3s5z", "3s5z_vs_3s6z", "3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "1c3s5z", "2c_vs_64zg"]:
            # Protoss maps have shields
            self.shield_bits_ally = 1
            self.shield_bits_enemy = 1
            
            if map_name == "1c3s5z":
                self.unit_type_bits = 3 # Colossus, Stalker, Zealot
            else:
                self.unit_type_bits = 2 # Stalker, Zealot
        elif map_name in ["MMM", "MMM2"]:
            self.unit_type_bits = 3 # Marine, Marauder, Medivac
            # Medivacs explicitly handled by energy (in cooldown slot)
        
        # 特征数量跨度
        self.n_feats_per_agent = 4 + self.shield_bits_ally + self.unit_type_bits
        self.n_feats_per_enemy = 3 + self.shield_bits_enemy + self.unit_type_bits

    def describe_state(self, obs_list: list, global_state: np.ndarray = None) -> str:
        """
        Generates a text description of the current battle state.
        
        Args:
            obs_list: List of observations for each agent.
            global_state: Optional global state vector.
            
        Returns:
            A string describing the situation.
        """
        description = []
        description.append(f"Battle with {self.n_agents} allies and {self.n_enemies} enemies.")
        
        # === 解析 Global State 获取详细信息 ===
        if global_state is not None:
            try:
                alive_allies = []

                # 1. 解析盟友信息 (Allies)
                description.append("\n[Ally Status]")
                for i in range(self.n_agents):
                    start_idx = i * self.n_feats_per_agent
                    
                    health = global_state[start_idx]
                    cd_or_energy = global_state[start_idx + 1]
                    pos_x = global_state[start_idx + 2]
                    pos_y = global_state[start_idx + 3]
                    
                    status = "Alive" if health > 0 else "Dead"
                    
                    if health > 0:
                        desc_str = f"- Agent {i}: {status}, Health: {health:.2f}, Pos: ({pos_x:.2f}, {pos_y:.2f})"
                        alive_allies.append({"id": i, "health": float(health), "x": float(pos_x), "y": float(pos_y)})
                        
                        ind = 4
                        # 读取护盾
                        if self.shield_bits_ally > 0:
                            shield = global_state[start_idx + ind]
                            desc_str += f", Shield: {shield:.2f}"
                            ind += 1
                        
                        # 读取单位类型
                        if self.unit_type_bits > 0:
                            type_feats = global_state[start_idx + ind : start_idx + ind + self.unit_type_bits]
                            type_idx = np.argmax(type_feats) if np.sum(type_feats) > 0 else -1
                            
                            # 根据地图做粗略映射，让 LLM 知道兵种
                            type_name = f"TypeID_{type_idx}"
                            if "s" in self.map_name and "z" in self.map_name: # 比如 3s5z
                                type_name = "Stalker" if type_idx == 0 else "Zealot"
                            elif "MMM" in self.map_name:
                                type_name = ["Marine", "Marauder", "Medivac"][type_idx] if 0 <= type_idx <= 2 else type_name
                                
                            desc_str += f", Unit: {type_name}"
                            
                        description.append(desc_str)
                    else:
                        description.append(f"- Agent {i}: Dead")

                # 1.1 基于全队血量与相对差值的角色候选分析
                description.append("\n[HP Analysis]")
                if alive_allies:
                    hp_values = np.array([a["health"] for a in alive_allies], dtype=np.float32)
                    hp_mean = float(np.mean(hp_values))
                    hp_min = float(np.min(hp_values))
                    hp_max = float(np.max(hp_values))
                    low_hp_exists = hp_min < self.low_hp_threshold

                    sorted_alive = sorted(alive_allies, key=lambda x: x["health"])
                    lowest_two = sorted_alive[:2]
                    lowest_two_ids = [a["id"] for a in lowest_two]

                    survivor_candidates = []
                    shield_candidates = []

                    if low_hp_exists:
                        for a in lowest_two:
                            if (hp_max - a["health"]) >= self.survivor_gap_threshold:
                                survivor_candidates.append(a["id"])

                        if lowest_two:
                            for a in alive_allies:
                                if a["id"] in lowest_two_ids:
                                    continue
                                if (a["health"] - hp_min) < self.shield_gap_threshold:
                                    continue

                                nearest_low_dist = min(
                                    float(np.hypot(a["x"] - low["x"], a["y"] - low["y"])) for low in lowest_two
                                )
                                if nearest_low_dist <= self.protect_distance_threshold:
                                    shield_candidates.append(a["id"])

                    description.append(
                        f"- TeamHP: mean={hp_mean:.2f}, min={hp_min:.2f}, max={hp_max:.2f}, "
                        f"low_hp_exists={str(low_hp_exists)} (threshold={self.low_hp_threshold:.2f})"
                    )
                    description.append(
                        f"- Relative thresholds: survivor_gap={self.survivor_gap_threshold:.2f}, "
                        f"shield_gap={self.shield_gap_threshold:.2f}, protect_dist={self.protect_distance_threshold:.2f}"
                    )
                    description.append(f"- Lowest HP agents (up to 2): {lowest_two_ids}")
                    description.append(f"- Survivor candidates: {survivor_candidates}")
                    description.append(f"- Shield candidates: {shield_candidates}")
                else:
                    description.append("- No alive allies.")

                # 2. 解析敌方信息 (Enemies)
                description.append("\n[Enemy Status]")
                enemy_start_offset = self.n_agents * self.n_feats_per_agent
                
                for i in range(self.n_enemies):
                    start_idx = enemy_start_offset + (i * self.n_feats_per_enemy)
                    
                    if start_idx + 2 < len(global_state):
                        health = global_state[start_idx]
                        pos_x = global_state[start_idx + 1]
                        pos_y = global_state[start_idx + 2]
                        
                        if health > 0:
                            desc_str = f"- Enemy {i}: Health: {health:.2f}, Pos: ({pos_x:.2f}, {pos_y:.2f})"
                            
                            ind = 3
                            # 读取护盾
                            if self.shield_bits_enemy > 0:
                                shield = global_state[start_idx + ind]
                                desc_str += f", Shield: {shield:.2f}"
                                ind += 1
                                
                            # 读取单位类型
                            if self.unit_type_bits > 0:
                                type_feats = global_state[start_idx + ind : start_idx + ind + self.unit_type_bits]
                                type_idx = np.argmax(type_feats) if np.sum(type_feats) > 0 else -1
                                
                                type_name = f"TypeID_{type_idx}"
                                if "s" in self.map_name and "z" in self.map_name:
                                    type_name = "Stalker" if type_idx == 0 else "Zealot"
                                elif "MMM" in self.map_name:
                                    type_name = ["Marine", "Marauder", "Medivac"][type_idx] if 0 <= type_idx <= 2 else type_name
                                    
                                desc_str += f", Unit: {type_name}"
                                
                            description.append(desc_str)
            except Exception as e:
                print(f"[Warning] Failed to parse global state: {e}")
                description.append("(Detailed unit stats unavailable due to state format mismatch)")
        else:
            description.append("Current Status:")
            for i in range(self.n_agents):
                description.append(f"- Agent {i}: Alive, ready for orders.")
            
        return "\n".join(description)

    def describe_from_env_info(self, env_info: dict) -> str:
        """
        Alternative method if the environment provides a structured info dict.
        """
        # If you can modify the env to return a dict of unit statuses, use that
