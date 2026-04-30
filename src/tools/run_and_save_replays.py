#!/usr/bin/env python3
"""
Run a saved model for a number of episodes and save SC2 replays.

Defaults target the attached model folder 5001550 and map 5m_vs_6m.

Usage (from repository root):
  python3 src/tools/run_and_save_replays.py \
    --model-path results/models/qmix_env=4_adam_td_lambda__2026-04-13_14-22-26/5001550 \
      --n-episodes 10 \
      --replay-dir qmix-4-5001550/replays

The script will build the same network architecture used for QMIX (n_mac / n_rnn)
and load `agent.th`, `mixer.th` and `opt.th` from the model folder.
Replays will be written to `--replay-dir` (absolute path internally).
"""
from __future__ import annotations

import os
import sys
import argparse
from types import SimpleNamespace as SN
import json
import yaml


def main():
    parser = argparse.ArgumentParser()
    repo_src_default = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model = os.path.join(os.path.dirname(repo_src_default),
                                 "results",
                                 "models",
                                 "qmix_env=4_adam_td_lambda__2026-04-13_14-22-26",
                                 "5001550")

    parser.add_argument("--model-path", type=str, default=default_model,
                        help="Path to folder containing agent.th, mixer.th, opt.th")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to run and save")
    parser.add_argument("--map", type=str, default="5m_vs_6m", help="SMAC map to use")
    parser.add_argument("--replay-dir", type=str, default=os.path.join(os.path.dirname(repo_src_default), "replays"),
                        help="Directory to save replay files")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--alg-config", type=str, default="qmix", help="Algorithm config name under src/config/algs")
    parser.add_argument("--env-config", type=str, default="sc2", help="Env config name under src/config/envs")
    args = parser.parse_args()

    def recursive_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def load_yaml(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def infer_checkpoint_map(repo_root, model_parent_dir):
        sacred_root = os.path.join(repo_root, "results", "sacred")
        if not os.path.isdir(sacred_root):
            return None
        needle = f"results/models/{model_parent_dir}/"
        for map_name in os.listdir(sacred_root):
            map_dir = os.path.join(sacred_root, map_name)
            if not os.path.isdir(map_dir):
                continue
            for algo_dir in os.listdir(map_dir):
                algo_path = os.path.join(map_dir, algo_dir)
                if not os.path.isdir(algo_path):
                    continue
                for run_id in os.listdir(algo_path):
                    run_dir = os.path.join(algo_path, run_id)
                    cout_path = os.path.join(run_dir, "cout.txt")
                    if not os.path.isfile(cout_path):
                        continue
                    try:
                        with open(cout_path, "r", errors="ignore") as f:
                            text = f.read()
                        if needle in text:
                            return map_name
                    except Exception:
                        pass
        return None

    # Ensure 'src' is on sys.path so top-level imports like 'learners' work
    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_pkg = os.path.dirname(src_dir)
    if src_pkg not in sys.path:
        sys.path.insert(0, src_pkg)

    # Local imports (now that src is on path)
    import torch as th
    from types import SimpleNamespace
    from utils.logging import get_logger, Logger
    from runners import REGISTRY as r_REGISTRY
    from controllers import REGISTRY as mac_REGISTRY
    from learners import REGISTRY as le_REGISTRY
    from components.episode_buffer import ReplayBuffer
    from components.transforms import OneHot

    # Build cfg by reading config files exactly like main.py would do.
    cfg_root = os.path.join(src_pkg, "config")
    default_cfg = load_yaml(os.path.join(cfg_root, "default.yaml"))
    env_cfg = load_yaml(os.path.join(cfg_root, "envs", f"{args.env_config}.yaml"))
    alg_cfg = load_yaml(os.path.join(cfg_root, "algs", f"{args.alg_config}.yaml"))
    cfg = recursive_dict_update(default_cfg, env_cfg)
    cfg = recursive_dict_update(cfg, alg_cfg)

    # Try to recover exact training config from sacred as a higher-priority source.
    model_path = os.path.abspath(args.model_path)
    model_parent = os.path.basename(os.path.dirname(model_path))
    algo_name = model_parent.split("__")[0]
    repo_root = os.path.dirname(repo_src_default)
    sacred_base = os.path.join(repo_root, "results", "sacred", args.map, algo_name)
    if os.path.isdir(sacred_base):
        run_dirs = [d for d in os.listdir(sacred_base) if d.isdigit() and os.path.isdir(os.path.join(sacred_base, d))]
        if run_dirs:
            run_choice = str(min(map(int, run_dirs)))
            sacred_cfg = os.path.join(sacred_base, run_choice, "config.json")
            if os.path.exists(sacred_cfg):
                try:
                    with open(sacred_cfg, "r") as f:
                        trained_cfg = json.load(f)
                    cfg.update(trained_cfg)
                    print(f"Loaded sacred config: {sacred_cfg}")
                except Exception as exc:
                    print(f"Warning: failed to load sacred config {sacred_cfg}: {exc}")

    # Force evaluation overrides after merging config.
    cfg["runner"] = "episode"
    cfg["batch_size_run"] = 1
    cfg["test_nepisode"] = args.n_episodes
    cfg["evaluate"] = True
    cfg["save_replay"] = True
    cfg["use_cuda"] = bool(args.use_cuda and th.cuda.is_available())
    cfg["env"] = "sc2"
    cfg.setdefault("env_args", {})
    cfg["env_args"]["map_name"] = args.map
    cfg["env_args"]["replay_dir"] = os.path.abspath(args.replay_dir)
    cfg["env_args"]["replay_prefix"] = "eval"

    # Helpful hint: infer original map from sacred logs for this model folder.
    inferred_map = infer_checkpoint_map(repo_root, model_parent)
    if inferred_map is not None and inferred_map != args.map:
        print("Warning: checkpoint was trained on map '{}' but current --map is '{}'".format(inferred_map, args.map))

    # Create args object
    args_ns = SN(**cfg)
    args_ns.device = "cuda" if args_ns.use_cuda else "cpu"

    # Logger
    console_logger = get_logger()
    logger = Logger(console_logger)

    # Make replay dir
    os.makedirs(args_ns.env_args["replay_dir"], exist_ok=True)

    # Init runner
    runner = r_REGISTRY[args_ns.runner](args=args_ns, logger=logger)
    env_info = runner.get_env_info()

    # Fill in env-dependent args
    args_ns.n_agents = env_info["n_agents"]
    args_ns.n_actions = env_info["n_actions"]
    args_ns.state_shape = env_info["state_shape"]

    # Scheme / buffer used only to construct MAC
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args_ns.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args_ns.n_actions)])}

    buffer = ReplayBuffer(scheme, groups, args_ns.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args_ns.buffer_cpu_only else args_ns.device)

    # Build MAC and attach to runner
    mac = mac_REGISTRY[args_ns.mac](buffer.scheme, groups, args_ns)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Build learner and load model
    learner = le_REGISTRY[args_ns.learner](mac, buffer.scheme, logger, args_ns)
    if args_ns.use_cuda:
        learner.cuda()

    # Resolve model path: if user provided parent dir with numeric subdirs, pick max
    if os.path.isdir(model_path):
        # If contains agent.th directly, assume this is the model folder
        if os.path.exists(os.path.join(model_path, "agent.th")):
            load_path = model_path
            try:
                runner.t_env = int(os.path.basename(model_path))
            except Exception:
                runner.t_env = 0
        else:
            # Look for numeric subdirectories and pick max
            subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)) and d.isdigit()]
            if len(subdirs) == 0:
                raise FileNotFoundError(f"No numeric checkpoints found in {model_path} and no agent.th present.")
            chosen = max(map(int, subdirs))
            load_path = os.path.join(model_path, str(chosen))
            runner.t_env = chosen
    else:
        raise FileNotFoundError(f"Model path {model_path} not found")

    print(f"Loading models from: {load_path}")
    try:
        learner.load_models(load_path)
    except RuntimeError as exc:
        print("Failed to load checkpoint due to architecture mismatch.")
        if inferred_map is not None:
            print("Detected checkpoint map: {}".format(inferred_map))
        print("Use a checkpoint trained on the same map as --map, or run with --map set to the checkpoint map.")
        raise

    # Run episodes and save replay per-episode
    n = int(cfg.get("test_nepisode", args.n_episodes))
    for i in range(n):
        print(f"Running episode {i+1}/{n}...")
        # Give a unique prefix per episode
        runner.env.replay_prefix = f"model_{os.path.basename(load_path)}_ep{i+1}"
        runner.run(test_mode=True)
        runner.save_replay()

    runner.close_env()
    print(f"Done. Replays written to {args_ns.env_args['replay_dir']}")


if __name__ == '__main__':
    main()
