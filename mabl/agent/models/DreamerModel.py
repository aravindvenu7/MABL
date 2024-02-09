import torch
import torch.nn as nn

from environments import Env
from networks.dreamer.dense import DenseBinaryModel, DenseModel
from networks.dreamer.vae import Encoder, Decoder
from networks.dreamer.rnns import RSSMRepresentation, RSSMTransition
from configs.dreamer.DreamerAgentConfig import RSSMState

class DreamerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.action_size = config.ACTION_SIZE

        self.observation_encoder = Encoder(in_dim=config.IN_DIM, hidden=config.HIDDEN, embed=config.EMBED)
        self.observation_decoder = Decoder(embed=config.FEAT, hidden=config.HIDDEN, out_dim=config.IN_DIM)

        self.transition = RSSMTransition(config, config.MODEL_HIDDEN)
        self.representation = RSSMRepresentation(config, self.transition)
        self.reward_model = DenseModel(config.FEAT, 1, config.REWARD_LAYERS, config.REWARD_HIDDEN)
        #self.g_reward_model = DenseModel(config.GLOBAL_FEAT, 1, config.REWARD_LAYERS, config.REWARD_HIDDEN)
        ##############################################################################################
        #self.state_reward_model = DenseModel(config.GLOBAL_FEAT, 1, config.REWARD_LAYERS, config.REWARD_HIDDEN)
        ################################################################################################
        self.pcont = DenseBinaryModel(config.FEAT+config.GLOBAL_FEAT, 1, config.PCONT_LAYERS, config.PCONT_HIDDEN)

        if config.ENV_TYPE == Env.STARCRAFT:
            self.av_action = DenseBinaryModel(config.FEAT+config.GLOBAL_FEAT, config.ACTION_SIZE, config.PCONT_LAYERS, config.PCONT_HIDDEN)
        else:
            self.av_action = None

        self.q_features = DenseModel(config.HIDDEN, config.PCONT_HIDDEN, 1, config.PCONT_HIDDEN)
        self.q_action = nn.Linear(config.PCONT_HIDDEN, config.ACTION_SIZE)

        self.state_encoder = Encoder(in_dim=config.STATE_DIM, hidden=config.HIDDEN, embed=config.STATE_EMBED)
        #self.attention_stack = AttentionEncoder(3, config.STATE_EMBED, config.STATE_EMBED, dropout=0.1)
        self.state_decoder = Decoder(embed=config.GLOBAL_FEAT, hidden=config.HIDDEN, out_dim=config.STATE_DIM)

    def forward(self, observations, prev_actions=None, prev_states=None, mask=None):
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), observations.size(1), self.action_size,
                                       device=observations.device)

        if prev_states is None:
            prev_states = self.representation.initial_agent_state(prev_actions.size(0), observations.size(1),
                                                            device=observations.device)

        return self.get_state_representation(observations, prev_actions, prev_states, mask)

    def get_state_representation(self, observations, prev_actions, prev_states, mask):
        """
        :param observations: size(batch, n_agents, in_dim)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState
        """
        batch_size = 1
        n_agents = observations.shape[1]
        obs_embeds = self.observation_encoder(observations)
        #pass prev state and prev action through agent's prior rnn
        stoch_input = self.transition.agent_rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        deter_state = self.transition.agent_cell(stoch_input.reshape(1, n_agents, -1),
                                 prev_states.deter.reshape(1, n_agents, -1))[0].reshape(1, n_agents, -1)        
       
        #pass output of rnn and current obsrvation through agent's posterior network
        x_agent = torch.cat([deter_state, obs_embeds], dim=-1)
        agent_logits, agent_stoch_state = self.representation.agent_stochastic_posterior_model(x_agent)
        states = RSSMState(logits=agent_logits, stoch=agent_stoch_state, deter=deter_state)
        return states


