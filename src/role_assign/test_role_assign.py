import numpy as np
from role_manager import RoleManager
from role_config import RoleType

def test_integration():
    print("Testing Role Assignment Module...")
    
    # 1. 初始化管理器 (使用 Mock 或真实 API)
    # 如果您有真实 API Key，请填入 api_key="sk-..."
    manager = RoleManager(n_agents=3, n_enemies=3, update_interval=10, api_key="f2841229-111e-482b-8753-96079f62ee6a")
    
    # 2. 模拟环境数据
    obs_list = [np.zeros(10) for _ in range(3)] # 模拟观测
    
    # 3. 触发第一次更新 (Step 0)
    print("\n--- Step 0: Initial Update ---")
    manager.update_roles(current_step=0, obs_list=obs_list)
    
    # 打印当前角色
    for agent_id, role in manager.current_roles.items():
        print(f"Agent {agent_id}: {role.value}")
        
    # 4. 获取编码
    encodings = manager.get_all_encodings()
    print(f"\nEncodings shape: {encodings.shape}")
    print(f"Agent 0 encoding: {encodings[0]}")
    
    # 5. 模拟后续步骤 (Step 5 - 不应更新)
    print("\n--- Step 5: No Update Expected ---")
    manager.update_roles(current_step=5, obs_list=obs_list)
    
    # 6. 模拟更新步骤 (Step 10 - 应触发更新)
    print("\n--- Step 10: Update Expected ---")
    manager.update_roles(current_step=10, obs_list=obs_list)
    for agent_id, role in manager.current_roles.items():
        print(f"Agent {agent_id}: {role.value}")

if __name__ == "__main__":
    test_integration()