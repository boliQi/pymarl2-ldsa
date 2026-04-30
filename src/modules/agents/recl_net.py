"""
RECL Network from ACORM (Agent-role Coordination Learning).
Contains Agent_Embedding, Role_Embedding, and Role_Classifier networks.

Reference: ACORM-master/ACORM_QMIX/util/net.py and algorithm/acorm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent_Embedding(nn.Module):
    """
    Agent Embedding Network: (obs, last_action) -> agent_embedding
    Uses GRUCell for temporal processing.
    """
    def __init__(self, obs_dim, action_dim, agent_embedding_dim):
        super(Agent_Embedding, self).__init__()
        self.input_dim = obs_dim + action_dim
        self.agent_embedding_dim = agent_embedding_dim

        self.fc1 = nn.Linear(self.input_dim, agent_embedding_dim)
        self.rnn_hidden = None
        self.agent_embedding_fc = nn.GRUCell(agent_embedding_dim, agent_embedding_dim)
        self.fc2 = nn.Linear(agent_embedding_dim, agent_embedding_dim)

    def init_hidden(self):
        """Initialize hidden state"""
        self.rnn_hidden = None

    def forward(self, obs, last_a, detach=False):
        """
        Args:
            obs: (batch*n_agents, obs_dim)
            last_a: (batch*n_agents, action_dim)
            detach: whether to detach output from computation graph
        Returns:
            agent_embedding: (batch*n_agents, agent_embedding_dim)
        """
        inputs = torch.cat([obs, last_a], dim=-1)
        fc1_out = torch.relu(self.fc1(inputs))
        
        # Initialize hidden state if needed
        if self.rnn_hidden is None:
            self.rnn_hidden = torch.zeros(fc1_out.size(0), self.agent_embedding_dim, 
                                          device=fc1_out.device, dtype=fc1_out.dtype)
        
        self.rnn_hidden = self.agent_embedding_fc(fc1_out, self.rnn_hidden)
        fc2_out = self.fc2(self.rnn_hidden)
        
        if detach:
            return fc2_out.detach()
        return fc2_out


class Role_Embedding(nn.Module):
    """
    Role Embedding Network: agent_embedding -> role_embedding
    Maps agent embedding to role embedding space.
    """
    def __init__(self, agent_embedding_dim, role_embedding_dim, use_ln=False):
        super(Role_Embedding, self).__init__()
        self.agent_embedding_dim = agent_embedding_dim
        self.role_embedding_dim = role_embedding_dim
        self.use_ln = use_ln

        if use_ln:
            self.role_embedding = nn.ModuleList([
                nn.Linear(agent_embedding_dim, role_embedding_dim),
                nn.LayerNorm(role_embedding_dim)
            ])
        else:
            self.role_embedding = nn.Linear(agent_embedding_dim, role_embedding_dim)
    
    def forward(self, agent_embedding, detach=False):
        """
        Args:
            agent_embedding: (batch*n_agents, agent_embedding_dim)
            detach: whether to detach output
        Returns:
            role_embedding: (batch*n_agents, role_embedding_dim)
        """
        if self.use_ln:
            output = self.role_embedding[1](self.role_embedding[0](agent_embedding))
        else:
            output = self.role_embedding(agent_embedding)
        
        if detach:
            output = output.detach()
        
        output = torch.sigmoid(output)
        return output


class Agent_Embedding_Decoder(nn.Module):
    """
    Agent Embedding Decoder: agent_embedding -> (next_obs, agent_id)
    Used for pretraining agent embedding to reconstruct next observation.
    """
    def __init__(self, agent_embedding_dim, obs_dim, n_agents):
        super(Agent_Embedding_Decoder, self).__init__()
        self.agent_embedding_dim = agent_embedding_dim
        # Output: next_obs + agent_id_logits (if needed, ACORM uses agent_id)
        self.decoder_out_dim = obs_dim + n_agents    
        
        self.fc1 = nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim)
        self.fc2 = nn.Linear(self.agent_embedding_dim, self.decoder_out_dim)

    def forward(self, agent_embedding):
        fc1_out = torch.relu(self.fc1(agent_embedding))
        decoder_out = self.fc2(fc1_out)
        return decoder_out


class RECL_NET(nn.Module):
    """
    Role Embedding Contrastive Learning Network.
    Combines Agent_Embedding, Role_Embedding, and Decoder.
    """
    def __init__(self, obs_dim, action_dim, n_agents, agent_embedding_dim=128, 
                 role_embedding_dim=64, n_roles=4, use_ln=False):
        super(RECL_NET, self).__init__()
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_embedding_dim = agent_embedding_dim
        self.role_embedding_dim = role_embedding_dim
        self.n_roles = n_roles
        
        # Networks
        self.agent_embedding_net = Agent_Embedding(obs_dim, action_dim, agent_embedding_dim)
        self.role_embedding_net = Role_Embedding(agent_embedding_dim, role_embedding_dim, use_ln)
        self.agent_decoder = Agent_Embedding_Decoder(agent_embedding_dim, obs_dim, n_agents)

    def init_hidden(self):
        """Reset hidden states"""
        self.agent_embedding_net.init_hidden()

    def forward(self, obs, last_action, detach=False):
        """
        Main forward: obs + last_action -> role_embedding
        """
        agent_embedding = self.agent_embedding_net(obs, last_action, detach)
        role_embedding = self.role_embedding_net(agent_embedding)
        return role_embedding

    def agent_embedding_forward(self, obs, last_action, detach=False):
        """Compute only agent embedding"""
        return self.agent_embedding_net(obs, last_action, detach)

    def role_embedding_forward(self, agent_embedding, detach=False):
        """Compute role embedding from agent embedding"""
        return self.role_embedding_net(agent_embedding, detach)
    
    def decode_agent_embedding(self, agent_embedding):
        """Decode agent embedding for pretraining"""
        return self.agent_decoder(agent_embedding)

    def batch_role_embed_forward(self, batch_obs, batch_last_a, max_episode_len, detach=False):
        """
        Batch forward for entire episode.
        
        Args:
            batch_obs: (batch_size, max_episode_len+1, n_agents, obs_dim)
            batch_last_a: (batch_size, max_episode_len+1, n_agents, action_dim)
            max_episode_len: maximum episode length
            detach: whether to detach agent_embedding
        
        Returns:
            agent_embeddings: (batch_size, max_episode_len+1, n_agents, agent_embedding_dim)
            role_embeddings: (batch_size, max_episode_len+1, n_agents, role_embedding_dim)
        """
        batch_size = batch_obs.shape[0]
        
        # Reset hidden state
        self.agent_embedding_net.rnn_hidden = None
        
        agent_embeddings = []
        for t in range(max_episode_len + 1):
            # obs: (batch_size, n_agents, obs_dim) -> (batch_size*n_agents, obs_dim)
            obs_t = batch_obs[:, t].reshape(-1, self.obs_dim)
            last_a_t = batch_last_a[:, t].reshape(-1, self.action_dim)
            
            agent_embedding = self.agent_embedding_forward(obs_t, last_a_t, detach=detach)
            # Reshape: (batch_size*n_agents, dim) -> (batch_size, n_agents, dim)
            agent_embedding = agent_embedding.reshape(batch_size, self.n_agents, -1)
            agent_embeddings.append(agent_embedding)
        
        # Stack: (batch_size, max_episode_len+1, n_agents, agent_embedding_dim)
        agent_embeddings = torch.stack(agent_embeddings, dim=1)
        
        # Flatten for role embedding: (batch_size*(max_episode_len+1)*n_agents, agent_embedding_dim)
        agent_emb_flat = agent_embeddings.reshape(-1, self.agent_embedding_dim)
        role_emb_flat = self.role_embedding_forward(agent_emb_flat, detach=False)
        
        # Reshape back: (batch_size, max_episode_len+1, n_agents, role_embedding_dim)
        role_embeddings = role_emb_flat.reshape(batch_size, max_episode_len + 1, 
                                                 self.n_agents, self.role_embedding_dim)
        
        return agent_embeddings, role_embeddings

    def get_role_parameters(self):
        """Return role-related parameters for optimizer (agent_emb + role_emb)"""
        return (list(self.agent_embedding_net.parameters()) + 
                list(self.role_embedding_net.parameters()))
    
    def get_role_phase_parameters(self):
        """
        Return only role_embedding parameters for Phase 1 training.
        Used when adjusting clustering purely based on role projection without altering agent representations.
        """
        return list(self.role_embedding_net.parameters())

    def get_recl_parameters(self):
        """Return all RECL parameters for TD Loss optimizer"""
        return (list(self.agent_embedding_net.parameters()) + 
                list(self.role_embedding_net.parameters()))

    def get_agent_embedding_parameters(self):
        """Return parameters for agent embedding pretraining (encoder + decoder)"""
        return (list(self.agent_embedding_net.parameters()) +
                list(self.agent_decoder.parameters()))
