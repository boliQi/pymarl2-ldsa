import sys
import os
import numpy as np

# === [修复] 兼容旧版 SMAC 代码 ===
np.bool = bool
# ==============================

from smac.env import StarCraft2Env

# 设置你要测试的地图名称
MAP_NAME = "5m_vs_6m"  # <--- 在这里修改地图名，例如 '2s3z', '3m', '8m' 等

def analyze_state_structure(map_name):
    print(f"=== Analyzing Global State for Map: {map_name} ===\n")
    
    try:
        # 初始化环境
        env = StarCraft2Env(map_name=map_name)
        env_info = env.get_env_info()
        env.reset()
        state = env.get_state()
        
        n_agents = env.n_agents
        n_enemies = env.n_enemies
        
        print(f"Total State Size: {len(state)}")
        print(f"Number of Agents: {n_agents}")
        print(f"Number of Enemies: {n_enemies}")
        print("-" * 50)

        # === 核心逻辑：根据 SMAC 源码推导 State 结构 ===
        # SMAC 的 state 构建顺序通常是：
        # 1. Allies (Agents)
        # 2. Enemies
        # 3. Last Actions (如果开启)
        # 4. Timestep (如果开启)
        
        idx = 0
        
        # --- 1. 解析盟友 (Allies) ---
        print(f"\n[Allies Section] (Indices {idx} to ...)")
        
        # 计算每个盟友占用的特征数
        # 基础特征: Health(1) + Cooldown(1) + X(1) + Y(1) = 4
        ally_feat_size = 4 
        if env.shield_bits_ally > 0:
            ally_feat_size += 1 # Shield
        if env.unit_type_bits > 0:
            ally_feat_size += env.unit_type_bits # Unit Type (one-hot)
            
        print(f"Features per Ally: {ally_feat_size}")
        print(f"  - Health: 1")
        print(f"  - Cooldown: 1")
        print(f"  - X, Y: 2")
        if env.shield_bits_ally > 0: print(f"  - Shield: 1")
        if env.unit_type_bits > 0: print(f"  - Unit Type: {env.unit_type_bits}")

        for i in range(n_agents):
            start = idx
            end = idx + ally_feat_size
            values = state[start:end]
            print(f"  Agent {i} (Idx {start}-{end-1}): {np.round(values, 2)}")
            idx += ally_feat_size

        # --- 2. 解析敌人 (Enemies) ---
        print(f"\n[Enemies Section] (Indices {idx} to ...)")
        
        # 计算每个敌人占用的特征数
        # 基础特征: Health(1) + X(1) + Y(1) = 3 (敌人通常没有 Cooldown)
        enemy_feat_size = 3
        if env.shield_bits_enemy > 0:
            enemy_feat_size += 1
        if env.unit_type_bits > 0:
            enemy_feat_size += env.unit_type_bits
            
        print(f"Features per Enemy: {enemy_feat_size}")
        print(f"  - Health: 1")
        print(f"  - X, Y: 2")
        if env.shield_bits_enemy > 0: print(f"  - Shield: 1")
        if env.unit_type_bits > 0: print(f"  - Unit Type: {env.unit_type_bits}")

        for i in range(n_enemies):
            start = idx
            end = idx + enemy_feat_size
            # 防止越界（有些特殊地图可能没有某些信息）
            if end <= len(state):
                values = state[start:end]
                print(f"  Enemy {i} (Idx {start}-{end-1}): {np.round(values, 2)}")
                idx += enemy_feat_size

        # --- 3. 其他信息 ---
        if env.state_last_action:
            print(f"\n[Last Actions] (Indices {idx} to ...)")
            size = n_agents * env.n_actions
            print(f"  Size: {size} (One-hot actions for each agent)")
            idx += size
            
        if env.state_timestep_number:
            print(f"\n[Timestep] (Index {idx})")
            print(f"  Value: {state[idx]}")
            idx += 1

        env.close()
        
        print("\n" + "="*50)
        print("建议：根据上面的输出，更新 StateDescriptor 中的 n_feats_per_unit")
        print(f"Ally features start at index: 0")
        print(f"Enemy features start at index: {n_agents * ally_feat_size}")
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")
        print("请确保你已经安装了 SMAC 并且地图文件存在。")

if __name__ == "__main__":
    analyze_state_structure(MAP_NAME)