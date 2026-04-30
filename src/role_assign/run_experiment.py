import os
import subprocess
import sys
import argparse

class LLMRoleRunner:
    def __init__(self):
        # 获取当前脚本所在目录 (role_assign)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (llmrole)
        self.root_dir = os.path.dirname(current_dir)
        
        # 自动检测 pymarl 目录 (优先匹配 pymarl2，兼容 pymarl)
        if os.path.exists(os.path.join(self.root_dir, 'pymarl2')):
            self.pymarl_dir = os.path.join(self.root_dir, 'pymarl2')
        else:
            self.pymarl_dir = os.path.join(self.root_dir, 'pymarl')

    def run(self, args, extra_args):
        """
        构造并执行 PyMARL 训练命令，注入 LLM 参数
        """
        # 1. 设置环境变量
        env = os.environ.copy()
        
        # 解决多进程下 sklearn/numpy 的 OpenMP 死锁与卡顿问题
        env['OMP_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        env['VECLIB_MAXIMUM_THREADS'] = '1'
        env['NUMEXPR_NUM_THREADS'] = '1'
        
        # 关键：将项目根目录加入 PYTHONPATH，这样 pymarl 就能 import role_assign
        env['PYTHONPATH'] = self.root_dir 
        # 设置 GPU
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        # 2. 构造命令
        # 注意：我们将在 pymarl 目录下执行，所以 main.py 路径是 src/main.py
        cmd = [
            sys.executable, "src/main.py",
            f"--config={args.config}",      # 修改: 加等号
            "--env-config=sc2",             # 修改: 加等号
            "with"
        ]

        # 3. 注入 Sacred 配置参数
        sacred_params = [
            f"env_args.map_name={args.map}",
            f"seed={args.seed}",
            # === 关键结合点：将 LLM 参数传给 PyMARL ===
            f"role_update_interval={args.update_interval}",
            f"role_short_interval={args.short_interval}",
            f"role_warm_up_limit={args.warm_up_limit}",
            f"role_encoding_dim={args.role_encoding_dim}",
            f"use_big_model={args.big_model}",
            "use_llm_role=True"  # <--- 必须添加这一行，激活 PyMARL 中的钩子  
        ]

        # 如果提供了预训练 RECL 路径，覆盖 YAML 配置
        if args.pretrained_recl_path:
            sacred_params.append("load_pretrained_recl=True")
            sacred_params.append(f"pretrained_recl_path='{args.pretrained_recl_path}'")

        # 如果提供了 API Key，也传进去
        if args.api_key:
            sacred_params.append(f"llm_api_key='{args.api_key}'")

        # 合并参数
        cmd.extend(sacred_params)
        cmd.extend(extra_args)

        print(f"[*] Working Directory: {self.pymarl_dir}")
        print(f"[*] PYTHONPATH: {env['PYTHONPATH']}")
        print(f"[*] LLM Config: update_interval={args.update_interval}, short_interval={args.short_interval}, warm_up_limit={args.warm_up_limit}")
        print(f"[*] Command: {' '.join(cmd)}")
        print("-" * 50)

        # 4. 执行命令
        try:
            subprocess.run(cmd, cwd=self.pymarl_dir, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Training failed with error code {e.returncode}")
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-Role PyMARL Experiments")
    
    # 基础训练参数
    parser.add_argument("--map", type=str, default="5m_vs_6m", help="SC2 Map Name")
    parser.add_argument("--config", type=str, default="vdn", help="Algorithm Config (qmix, vdn, etc.)")
    parser.add_argument("--gpu", type=str, default="2", help="GPU ID")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    
    # === LLM 结合参数 ===
    parser.add_argument("--update_interval", type=int, default=10, help="Normal interval for LLM role updates (after warm_up)")
    parser.add_argument("--short_interval", type=int, default=5, help="Short interval for LLM role updates (during warm_up)")
    parser.add_argument("--warm_up_limit", type=int, default=20, help="Time step threshold to switch from short to normal interval")
    parser.add_argument("--api_key", type=str, default="f2841229-111e-482b-8753-96079f62ee6a", help="LLM API Key (optional)")
    parser.add_argument("--role_encoding_dim", type=int, default=4, help="role encoding dimension")
    parser.add_argument("--big_model", action="store_true", help="Use the big LLM model (e.g., doubao-seed)")
    parser.add_argument("--pretrained_recl_path", type=str, default="",
                        help="Path to pretrained RECL weights (relative to llmrole root). "
                             "Overrides YAML config pretrained_recl_path.")
    
    args, unknown = parser.parse_known_args()

    runner = LLMRoleRunner()
    runner.run(args, unknown)

    # python run_experiment.py --map 8m --config qmix t_max=2050000 batch_size_run=8
    # python run_experiment.py t_max=300 --interval 5 
    # python run_experiment.py --map 5m_vs_6m --interval 5 checkpoint_path="results/models/vdn_env=8_adam_td_lambda__2025-12-24_11-57-13" evaluate=True
# python run_experiment.py --map 5m_vs_6m --config vdn --seed 2
# CUDA_VISIBLE_DEVICES=1 python run_experiment.py --map 5m_vs_6m --config qmixrolellm --seed 2
# CUDA_VISIBLE_DEVICES=2 python run_experiment.py --map 5m_vs_6m --seed 7 --config qmixrolellm --big_model
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --map 3s5z --seed 1 --config qmixrolellm --big_model

# # 禁用 Role，使用自定义 WandB 名称
# python run_experiment.py \
#     --map 5m_vs_6m \
#     --config qmixrolellm \
#     use_role_predictor=False \
#     use_llm_role=False \
#     role_encoding_dim=0 \
#     use_wandb=True \
#     name="qmix_baseline_5m_vs_6m"

# python run_experiment.py \
#     --map 5m_vs_6m \
#     --config qmixrolellm \
#     use_role_predictor=True \
#     use_llm_role=False \
#     use_wandb=True \
#     wandb_project="qmix_comparison" \
#     name="qmix_role_mlp_5m_vs_6m_seed1"

# python run_experiment.py \
#     --map 5m_vs_6m \
#     --config qmixrolellm \
#     --seed 1 \
#     --interval 30 \
#     use_role_predictor=True \
#     use_llm_role=True \
#     use_wandb=True \
#     wandb_project="qmix_comparison" \
#     name="qmix_role_llm_5m_vs_6m_seed1"