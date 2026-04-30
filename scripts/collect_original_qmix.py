import os
import sys
import argparse
import pickle
import yaml
import torch as th
import numpy as np
from types import SimpleNamespace

class SimpleDummyLogger:
    def console_logger(self):
        import logging
        logger = logging.getLogger("OfflineDataCollection")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
        return logger

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_pymarl_config(args, base_dir):
    with open(os.path.join(base_dir, "src", "config", "default.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(base_dir, "src", "config", "envs", f"{args.env_config}.yaml"), "r") as f:
        env_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(base_dir, "src", "config", "algs", f"{args.config}.yaml"), "r") as f:
        alg_config = yaml.load(f, Loader=yaml.SafeLoader)
        
    config = recursive_dict_update(config, env_config)
    config = recursive_dict_update(config, alg_config)
    config["env_args"]["map_name"] = args.map
    return config

def collect_offline_data(args):
    # 动态插入 src 的路径，并获取当前主目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, "src"))
    
    from runners import REGISTRY as r_REGISTRY
    from controllers import REGISTRY as mac_REGISTRY
    from learners import REGISTRY as le_REGISTRY
    from components.transforms import OneHot
    from components.episode_buffer import ReplayBuffer
    from utils.logging import Logger

    # 1. 加载配置
    config_dict = get_pymarl_config(args, base_dir)
    config_dict["test_nepisode"] = args.n_episodes
    config_dict["checkpoint_path"] = args.checkpoint_path
    
    # 强制覆盖设置以进行数据收集
    config_dict["runner"] = "episode" 
    config_dict["batch_size_run"] = 1
    
    args_sn = SimpleNamespace(**config_dict)
    args_sn.device = "cuda" if args_sn.use_cuda else "cpu"
    
    # 2. 设置 Logger
    logger = Logger(SimpleDummyLogger().console_logger())
    
    # 3. 初始化 Runner
    runner = r_REGISTRY[args_sn.runner](args=args_sn, logger=logger)
    
    # 4. 获取环境信息并设置 scheme
    env_info = runner.get_env_info()
    args_sn.n_agents = env_info["n_agents"]
    args_sn.n_actions = env_info["n_actions"]
    args_sn.state_shape = env_info["state_shape"]
    args_sn.obs_shape = env_info["obs_shape"]
    
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args_sn.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args_sn.n_actions)])}
    
    buffer = ReplayBuffer(scheme, groups, args_sn.batch_size_run, env_info["episode_limit"] + 1, preprocess=preprocess, device="cpu")
    
    # 5. 初始化 MAC 和 Learner (原版 PyMARL 标准组件)
    mac = mac_REGISTRY[args_sn.mac](buffer.scheme, groups, args_sn)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    learner = le_REGISTRY[args_sn.learner](mac, scheme, logger, args_sn)
    if args_sn.use_cuda:
        learner.cuda()

    print(f"[*] Loading original QMIX checkpoint from: {args.checkpoint_path}")
    learner.load_models(args.checkpoint_path)
    
    collected_episodes = []
    print(f"[*] Starting offline data collection for {args.n_episodes} episodes...")
    
    for ep in range(args.n_episodes):
        batch = runner.run(test_mode=True)
        
        # 重新计算所有的 Q 值
        mac.init_hidden(batch.batch_size)
        mac_outs = []
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_outs.append(agent_outs)
        mac_outs = th.stack(mac_outs, dim=1)
        # 把 mac_outs 的时间维度切片到除去最后一步，与 actions/avail_actions 对齐
        mac_outs = mac_outs[:, :-1]
        
        actions = batch["actions"][:, :-1]
        chosen_action_qvals = th.gather(mac_outs, dim=3, index=actions).squeeze(3)
        
        avail_actions = batch["avail_actions"][:, :-1]
        mac_outs_masked = mac_outs.clone().detach()
        mac_outs_masked[avail_actions == 0] = -9999999
        max_action_qvals = mac_outs_masked.max(dim=3)[0]
        
        if hasattr(learner, "mixer") and learner.mixer is not None:
            states = batch["state"][:, :-1]
            tot_chosen_q = learner.mixer(chosen_action_qvals, states)
            tot_max_q = learner.mixer(max_action_qvals, states)
        else:
            tot_chosen_q = chosen_action_qvals.sum(dim=2, keepdim=True)
            tot_max_q = max_action_qvals.sum(dim=2, keepdim=True)
            
        episode_data = {
            "state": batch["state"][0].cpu().numpy(),
            "obs": batch["obs"][0].cpu().numpy(),
            "actions": batch["actions"][0].cpu().numpy(),
            "rewards": batch["reward"][0].cpu().numpy(),
            "terminated": batch["terminated"][0].cpu().numpy(),
            "avail_actions": batch["avail_actions"][0].cpu().numpy(),
            "q_vals_individual_chosen": chosen_action_qvals[0].detach().cpu().numpy(),
            "q_vals_individual_max": max_action_qvals[0].detach().cpu().numpy(),
            "q_vals_tot_chosen": tot_chosen_q[0].detach().cpu().numpy(),
            "q_vals_tot_max": tot_max_q[0].detach().cpu().numpy(),
        }
        
        collected_episodes.append(episode_data)
        if (ep + 1) % 10 == 0:
            print(f"Collected {ep + 1}/{args.n_episodes} episodes.")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(collected_episodes, f)
    
    print(f"[*] Successfully saved {len(collected_episodes)} episodes to {args.save_path}")
    runner.close_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="qmix_high_sample_efficiency", help="Name of the alg config (e.g., qmix)")
    parser.add_argument("--env-config", type=str, default="sc2", help="Name of the env config")
    parser.add_argument("--map", type=str, default="5m_vs_6m", help="Map name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint directory, e.g. results/models/...")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--save_path", type=str, default="offline_data_qmix.pkl", help="Where to save the collected dataset")
    args = parser.parse_args()
    
    collect_offline_data(args)
# python scripts/collect_original_qmix.py --config qmix_high_sample_efficiency --env-config sc2 --map 5m_vs_6m --checkpoint_path results/models/qmix_env=4_adam_td_lambda__2026-04-14_22-22-34/200104 --save_path datasets/qmix_env=4_adam_td_lambda__2026-04-14_22-22-34-200104.pkl --n_episodes 1000