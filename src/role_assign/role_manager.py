import numpy as np
from typing import List, Dict
from .role_config import RoleType, RoleConfig, RoleEncoder, SMAC_ROLES
from .llm_adapter import LLMAdapter
from .state_descriptor import StateDescriptor
import json
import os
import time
import fcntl  # 用于文件锁
import random

class RoleManager:
    def __init__(self, 
                 n_agents: int, 
                 n_enemies: int, 
                 update_interval: int = 50,
                 short_interval: int = 5,
                 warm_up_limit: int = 20,
                 api_key: str = None,
                 encoding_dim: int = 4,
                 map_name: str = "5m_vs_6m",
                 use_big_model: bool = False,
                 log_file: str = "llm_interaction_test.jsonl"):
        """
        Manages role assignments for agents.
        
        Args:
            n_agents: Number of agents
            n_enemies: Number of enemies
            update_interval: Normal LLM update interval (used after warm_up_limit)
            short_interval: Short interval for high-frequency updates at episode start
            warm_up_limit: Time step threshold after which to switch to normal interval
            api_key: LLM API key
            encoding_dim: Role encoding dimension
            log_file: Path to interaction log file
        """
        self.n_agents = n_agents
        self.update_interval = update_interval
        self.short_interval = short_interval
        self.warm_up_limit = warm_up_limit
        self.role_encoding_dim = encoding_dim
        self.api_key = api_key
        self.map_name = map_name
        self.use_big_model = use_big_model
        self.llm = None
        # self.llm = LLMAdapter(api_key=api_key)
        self.env_step = 0
        self.descriptor = StateDescriptor(n_agents, n_enemies, map_name=self.map_name)
        self.encoder = RoleEncoder(encoding_dim=encoding_dim)
        
        # === 新增: 初始化日志文件 (带时间戳) ===
        timestamp = time.strftime("%Y%m%d_%H%M")
        base_name, ext = os.path.splitext(log_file)
        self.log_file = f"{base_name}_{timestamp}{ext}"

        log_dir = os.path.dirname(os.path.abspath(self.log_file))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        print(f"[*] RoleManager logging to: {self.log_file}")
        # ==================================
        
        self.available_roles = list(SMAC_ROLES.values())
        # === [修复] 定义 available_role_types 供随机初始化使用 ===
        self.available_role_types = [cfg.role_type for cfg in self.available_roles]

        # === [核心修改] 将状态改为字典，支持多环境 ===
        # Key: env_id, Value: 该环境上一次更新的时间步
        self.env_last_update_step: Dict[int, int] = {}

          # === [优化] 日志缓冲区 ===
        self.log_buffer = []
        self.log_buffer_size = 5  # 每积攒 50 条写入一次
        
        # Key: env_id, Value: {agent_id: RoleType}
        self.env_roles: Dict[int, Dict[int, RoleType]] = {} 

     # === [新增] 延迟加载方法 ===
    def _get_llm(self):
        if self.llm is None:
            # 这一步只会在当前子进程第一次用到时执行一次
            self.llm = LLMAdapter(api_key=self.api_key, use_big_model=self.use_big_model)
        return self.llm
    # =========================
    
    def _init_env_state(self, env_id: int, current_step: int = None):
        """
        如果该环境是第一次运行或新Episode开始，初始化其状态
        
        Args:
            env_id: 环境ID
            current_step: 当前时间步。如果为0表示新Episode开始，强制重置；
                         如果为None则只在首次访问时初始化（安全检查）
        """
        should_init = False
        
        # 情况1: 新 Episode 开始时强制重置
        if current_step == 0:
            should_init = True
        # 情况2: 首次访问该 env_id 时初始化（安全检查，向后兼容）
        elif current_step is None and env_id not in self.env_last_update_step:
            should_init = True
        
        if should_init:
            # 设置为负数，确保第0步时 interval check (0 - (-999) >= 5) 为 True，立即触发 LLM
            self.env_last_update_step[env_id] = -999
            
            # === [修改] 随机分配初始角色，排除 DEAD ===
            active_roles = [r for r in self.available_role_types if r.name != "DEAD"]
            self.env_roles[env_id] = {
                i: random.choice(active_roles) 
                for i in range(self.n_agents)
            }
            
            initial_roles = [self.env_roles[env_id][i].name for i in range(self.n_agents)]
            print(f"[*] Initialized Env {env_id} with random roles: {initial_roles}")

    def _flush_log(self):
        """
        将缓冲区的数据强制写入文件 (带文件锁)
        """
        if not self.log_buffer:
            return

        try:
            # 使用 'a' 模式打开
            with open(self.log_file, "a", encoding="utf-8") as f:
                # [新增] 获取排他锁 (阻塞等待)
                fcntl.flock(f, fcntl.LOCK_EX)
                
                try:
                    for entry in self.log_buffer:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    # 强制刷新缓冲区到磁盘
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # [新增] 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            # 写入成功后清空缓冲区
            self.log_buffer = []
        except Exception as e:
            print(f"[!] Failed to write LLM log: {e}")
# ...existing code...

    def update_roles(self, current_step: int, obs_list: list, global_state: np.ndarray = None, env_id: int = 0, global_env_step: int = None):
        """
        Checks if roles need updating and updates them via LLM if so.
        
        Uses adaptive update frequency:
        - When current_step < warm_up_limit: use short_interval (high frequency)
        - When current_step >= warm_up_limit: use normal update_interval
        """
        if global_env_step is not None:
             self.env_step = global_env_step

        # 1. 确保该环境已初始化
        self._init_env_state(env_id, current_step)
        
        last_step = self.env_last_update_step[env_id]
        
        # === 自适应更新间隔 ===
        # Episode 初期使用 short_interval 高频更新，之后使用 update_interval
        if current_step < self.warm_up_limit:
            effective_interval = self.short_interval
        else:
            effective_interval = self.update_interval
        
        # print(f"[*] Env {env_id}: Step {current_step}, Last Update {last_step}, Interval {effective_interval}")

        if current_step - last_step >= effective_interval:
            state_desc = self.descriptor.describe_state(obs_list, global_state)
            
            interval_type = "short" if current_step < self.warm_up_limit else "normal"
            print(f"[*] Triggering LLM Role Update for Env {env_id} at step {current_step} (interval={interval_type}, freq={effective_interval})...")

            # 调用 LLM 获取角色分配
            current_roles = [self.env_roles[env_id][i].name for i in range(self.n_agents)]
            role_names = self._get_llm().get_role_assignment(
                state_description=state_desc,
                available_roles=self.available_roles,
                num_agents=self.n_agents,
                current_roles=current_roles,
                map_name=self.map_name
            )

            # 记录日志
            self._log_interaction(current_step, state_desc, role_names, env_id)
            
            # Map names back to RoleTypes
            name_to_type = {cfg.name.lower(): cfg.role_type for cfg in self.available_roles}
            
            for i, r_name in enumerate(role_names):
                # 清洗 r_name: 去除首尾空白和引号
                clean_name = r_name.strip().strip("'").strip('"')
                r_key = clean_name.lower()
                if r_key in name_to_type:
                    self.env_roles[env_id][i] = name_to_type[r_key]
                else:
                    print(f"[!] Invalid role '{r_name}' for agent {i}, keeping previous: {self.env_roles[env_id][i].name}")
                    time.sleep(5)          
            
            # [关键] 只更新当前环境的时间戳
            # # === [新增] 调试输出 ===
            # print(f"\n[DEBUG] After role assignment for Env {env_id}:")
            # print(f"  env_roles[{env_id}] type: {type(self.env_roles[env_id])}")
            # print(f"  Number of agents: {len(self.env_roles[env_id])}")
            
            # for i in range(self.n_agents):
            #     role = self.env_roles[env_id][i]
            #     encoding = self.encoder.encode(role)
            #     print(f"  Agent {i}: {role.name} (value={role.value}, type={type(role).__name__})")
            #     print(f"    Encoding: {encoding} (dtype={encoding.dtype})")
            # print()
            # # =======================
            self.env_last_update_step[env_id] = current_step

    def get_role_encoding(self, agent_id: int, env_id: int = 0) -> np.ndarray:
        """
        Returns the encoding vector for the agent's current role in specific env.
        """
        self._init_env_state(env_id) # 安全检查
        role_type = self.env_roles[env_id].get(agent_id, RoleType.SHIELD)
        return self.encoder.encode(role_type)

    def get_all_encodings(self, env_id: int = 0) -> np.ndarray:
        """
        Returns a batch of encodings for all agents in specific env.
        """
        # 测试输出结果
        encodings = np.array([self.get_role_encoding(i, env_id) for i in range(self.n_agents)])
        # print(f"[*] Env {env_id} Role Encodings:\n{encodings}")
        return encodings
    
    def _log_interaction(self, step, input_desc, output_roles, env_id):
        """
        将交互记录写入缓冲区，满额后写入文件
        """
        log_entry = {
            "env_id": env_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "env_step": self.env_step,
            "input_state_description": input_desc,
            "output_role_assignment": output_roles
        }
        
        self.log_buffer.append(log_entry)

        # 如果缓冲区满了，执行写入
        if len(self.log_buffer) >= self.log_buffer_size:
            self._flush_log()
        