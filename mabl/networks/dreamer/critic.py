import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model
from networks.transformer.layers import AttentionEncoder
import torch

class Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, activation=nn.ELU):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.feedforward_model = build_model(in_dim, 1, layers, hidden_size, activation)

    def forward(self, state_features, actions):
        return self.feedforward_model(state_features)


class MADDPGCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, global_feat = False, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, 1, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)
        self.prior = build_model(in_dim, 1, 3, hidden_size, activation)
        self.global_feat = global_feat
    def forward(self, agent_state_features, global_state_features, actions):
        
        n_agents = agent_state_features.shape[-2]
        batch_size = agent_state_features.shape[:-2]
        all_feat = torch.cat([agent_state_features, global_state_features], dim = -1)
        if(self.global_feat):
          global_embeds = F.relu(self.embed(all_feat))
          global_embeds = global_embeds.view(-1, n_agents, global_embeds.shape[-1])
          attn_embeds = F.relu(self._attention_stack(global_embeds).view(*batch_size, n_agents, global_embeds.shape[-1])) #IS ATTENTION NECESSARY HERE?
        else:
          agent_embeds = F.relu(self.embed(agent_state_features))
          agent_embeds = agent_embeds.view(-1, n_agents, agent_embeds.shape[-1])
          attn_embeds = F.relu(self._attention_stack(agent_embeds).view(*batch_size, n_agents, agent_embeds.shape[-1]))
          
        return self.feedforward_model(attn_embeds)

