import copy
import os
import numpy as np
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class LDSALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.use_recl_analysis = getattr(args, "use_recl_analysis", False)
        self.role_manager = None
        self.analysis_save_dir = None
        if self.use_recl_analysis:
            self.analysis_save_dir = os.path.join(
                getattr(args, "local_results_path", "results"),
                "embedding_vis",
                getattr(args, "unique_token", "ldsa"),
            )
            os.makedirs(self.analysis_save_dir, exist_ok=True)
            try:
                from role_assign.role_manager import RoleManager
                self.role_manager = RoleManager(
                    n_agents=args.n_agents,
                    n_enemies=max(1, getattr(args, "n_enemies", 1)),
                    update_interval=getattr(args, "role_update_interval", 10),
                    short_interval=getattr(args, "role_short_interval", 5),
                    warm_up_limit=getattr(args, "role_warm_up_limit", 20),
                    api_key=getattr(args, "llm_api_key", None),
                    encoding_dim=getattr(args, "role_encoding_dim", args.n_subtasks),
                    map_name=args.env_args.get("map_name", "5m_vs_6m"),
                    use_big_model=getattr(args, "use_big_model", False),
                    log_file=os.path.join(getattr(args, "local_results_path", "results"), "llm_interaction_train.jsonl"),
                )
            except Exception as exc:
                self.logger.console_logger.info("RoleManager init failed: {}".format(exc))
                self.role_manager = None

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.training_steps += 1
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]  # [bs, eplen, 1]
        actions = batch["actions"][:, :-1] # [bs, eplen, n_agents, 1]
        terminated = batch["terminated"][:, :-1].float() # [bs, eplen, 1]
        mask = batch["filled"][:, :-1].float() # [bs, eplen, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # [bs, eplen+1, n_agents, n_actions]

        # Calculate estimated Q-Values
        mac_out = []
        subtask_prob_logits = []
        subtask_prob_logits_last = []
        subtask_embeds = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, subtask_prob_logit, subtask_embed = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if t > 0:
                subtask_prob_logits.append(subtask_prob_logit)
            if t < batch.max_seq_length - 1:
                subtask_prob_logits_last.append(subtask_prob_logit)
                subtask_embeds.append(subtask_embed)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time [bs, eplen+1, n_agents, n_actions]
        subtask_prob_logits = th.stack(subtask_prob_logits, dim=1) # [bs, eplen, n_agents, n_subtasks]
        subtask_prob_logits_last = th.stack(subtask_prob_logits_last, dim=1) # [bs, eplen, n_agents, n_subtasks]
        subtask_embeds = th.stack(subtask_embeds, dim=1) # [bs, eplen, n_subtasks, embed_dim]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim [bs, eplen, n_agents]

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1]) # [bs, eplen, 1]
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # MSE loss of representation between two different subtasks
        subtask_embeds1 = subtask_embeds.unsqueeze(3) # [bs, eplen, n_subtasks, 1, embed_dim]
        subtask_embeds2 = subtask_embeds.unsqueeze(2).clone().detach() # [bs, eplen, 1, n_subtasks, embed_dim]
        subtask_dis = ((subtask_embeds1 - subtask_embeds2) ** 2).sum(dim=4, keepdim=True) # [bs, eplen, n_subtasks, n_subtasks, 1]
        subtask_dis = subtask_dis.sum([4, 3, 2]).unsqueeze(-1) # [bs, eplen, 1]
        subtask_dis = subtask_dis / (self.args.n_subtasks * (self.args.n_subtasks - 1)) # [bs, eplen, 1]
        masked_subtask_dis = subtask_dis * mask
        subtask_dis_loss = masked_subtask_dis.sum() / mask.sum()

        # KL loss of subtask prob between two adjacent frames
        subtask_probs = F.softmax(subtask_prob_logits, dim=-1) # [bs, eplen, n_agents, n_subtasks]
        subtask_probs_last = F.softmax(subtask_prob_logits_last, dim=-1) # [bs, eplen, n_agents, n_subtasks]
        subtask_prob_kl = th.sum(subtask_probs_last.detach() * ( - th.log(subtask_probs + 1e-8)), dim=[3, 2]).unsqueeze(-1) / self.args.n_agents #[bs, eplen, 1]
        mask_ = mask[:, 1:] # [bs, eplen-1, 1]
        mask_ = th.cat([mask_, th.zeros(mask_.shape[0], 1, 1, device=mask_.device)], dim=1) # [bs, eplen, 1]
        subtask_prob_kl_loss = subtask_prob_kl.sum() / mask_.sum()

        loss = td_loss - self.args.lambda_subtask_repr * subtask_dis_loss + self.args.lambda_subtask_prob * subtask_prob_kl_loss

        self._maybe_run_recl_analysis(batch, subtask_prob_logits, t_env)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("subtask_dis_loss", subtask_dis_loss.item(), t_env)
            self.logger.log_stat("subtask_prob_kl_loss", subtask_prob_kl_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _maybe_run_recl_analysis(self, batch: EpisodeBatch, subtask_prob_logits: th.Tensor, t_env: int):
        if not self.use_recl_analysis:
            return
        interval = max(1, getattr(self.args, "train_recl_freq", 100))
        if self.training_steps % interval != 0:
            return
        if self.role_manager is None or not hasattr(self.mac.agent, "recl_net") or self.mac.agent.recl_net is None:
            return

        with th.no_grad():
            obs = batch["obs"][:, :-1]
            actions_onehot = batch["actions_onehot"][:, :-1]
            zeros = th.zeros(actions_onehot.shape[0], 1, actions_onehot.shape[2], actions_onehot.shape[3], device=actions_onehot.device)
            last_actions = th.cat([zeros, actions_onehot[:, :-1]], dim=1)
            max_ep_t = obs.shape[1]
            _, recl_role_embeddings = self.mac.agent.recl_net.batch_role_embed_forward(obs, last_actions, max_ep_t - 1, detach=True)
            llm_labels = self._build_llm_labels(batch, t_env)
            if llm_labels is None:
                return
            self._log_consistency_metrics(recl_role_embeddings[:, :-1], subtask_prob_logits, llm_labels, batch["filled"][:, :-1], t_env)

    def _build_llm_labels(self, batch: EpisodeBatch, t_env: int):
        if self.role_manager is None:
            return None
        bs = batch.batch_size
        seq_len = batch.max_seq_length - 1
        n_agents = self.args.n_agents
        labels = th.full((bs, seq_len, n_agents), -1, dtype=th.long, device=batch.device)
        filled = batch["filled"][:, :-1].squeeze(-1)
        obs = batch["obs"][:, :-1].detach().cpu().numpy()
        states = batch["state"][:, :-1].detach().cpu().numpy()
        from role_assign.role_config import SMAC_ROLES
        role_to_id = {cfg.role_type: cfg.role_id for cfg in SMAC_ROLES.values()}
        for b in range(bs):
            self.role_manager.env_last_update_step.pop(b, None)
            self.role_manager.env_roles.pop(b, None)
            for t in range(seq_len):
                if filled[b, t].item() <= 0:
                    continue
                obs_list = [obs[b, t, a] for a in range(n_agents)]
                self.role_manager.update_roles(t, obs_list=obs_list, global_state=states[b, t], env_id=b, global_env_step=t_env)
                for a in range(n_agents):
                    role_type = self.role_manager.env_roles[b][a]
                    labels[b, t, a] = role_to_id.get(role_type, -1)
        return labels

    def _log_consistency_metrics(self, recl_embed, ldsa_logits, llm_labels, filled, t_env: int):
        valid = (filled.squeeze(-1) > 0) & (llm_labels >= 0)
        if valid.sum().item() < 8:
            return
        recl_flat = recl_embed[valid]
        ldsa_flat = ldsa_logits[valid]
        llm_flat = llm_labels[valid]
        recl_pred = self._nearest_centroid_labels(recl_flat, llm_flat, self.args.n_subtasks)
        ldsa_pred = ldsa_flat.argmax(dim=-1)

        acc_recl_llm = (recl_pred == llm_flat).float().mean().item()
        acc_ldsa_llm = (ldsa_pred == llm_flat).float().mean().item()
        acc_recl_ldsa = (recl_pred == ldsa_pred).float().mean().item()
        self.logger.log_stat("consistency/acc_recl_llm", acc_recl_llm, t_env)
        self.logger.log_stat("consistency/acc_ldsa_llm", acc_ldsa_llm, t_env)
        self.logger.log_stat("consistency/acc_recl_ldsa", acc_recl_ldsa, t_env)

        # PCA 2D projection stats
        recl_2d = self._pca2d(recl_flat)
        ldsa_2d = self._pca2d(ldsa_flat)
        self.logger.log_stat("consistency/pca_recl_var_x", recl_2d[:, 0].var().item(), t_env)
        self.logger.log_stat("consistency/pca_ldsa_var_x", ldsa_2d[:, 0].var().item(), t_env)

        # Structural similarity via representational dissimilarity matrices
        take = min(256, recl_flat.shape[0])
        idx = th.randperm(recl_flat.shape[0], device=recl_flat.device)[:take]
        recl_rdm = self._rdm(recl_flat[idx])
        ldsa_rdm = self._rdm(ldsa_flat[idx].float())
        label_rdm = self._label_rdm(llm_flat[idx])
        self.logger.log_stat("consistency/rdm_corr_recl_ldsa", self._corrcoef(recl_rdm, ldsa_rdm).item(), t_env)
        self.logger.log_stat("consistency/rdm_corr_recl_llm", self._corrcoef(recl_rdm, label_rdm).item(), t_env)
        self.logger.log_stat("consistency/rdm_corr_ldsa_llm", self._corrcoef(ldsa_rdm, label_rdm).item(), t_env)

    def _nearest_centroid_labels(self, embeddings, labels, n_classes):
        centroids = []
        for c in range(n_classes):
            mask = labels == c
            if mask.any():
                centroids.append(F.normalize(embeddings[mask].mean(dim=0), dim=0))
            else:
                centroids.append(th.zeros(embeddings.shape[1], device=embeddings.device))
        centroids = th.stack(centroids, dim=0)
        sims = th.matmul(F.normalize(embeddings, dim=-1), centroids.t())
        return sims.argmax(dim=-1)

    def _pca2d(self, x):
        x = x.float()
        x = x - x.mean(dim=0, keepdim=True)
        try:
            _, _, v = th.pca_lowrank(x, q=min(2, x.shape[1]))
            return x @ v[:, :2]
        except Exception:
            return x[:, :2] if x.shape[1] >= 2 else th.cat([x, th.zeros_like(x)], dim=1)

    def _rdm(self, x):
        x = F.normalize(x.float(), dim=-1)
        sim = th.matmul(x, x.t()).clamp(-1, 1)
        return (1 - sim).reshape(-1)

    def _label_rdm(self, y):
        y = y.view(-1, 1)
        return (y != y.t()).float().reshape(-1)

    def _corrcoef(self, a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.std() * b.std()) + 1e-8
        return (a * b).mean() / denom

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
