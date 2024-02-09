import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical

from configs.dreamer.DreamerAgentConfig import RSSMState
from networks.transformer.layers import AttentionEncoder


def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU):
        super().__init__()
        self.config = config
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self.global_stoch_size = config.GLOBAL_STOCHASTIC
        self.global_deter_size = config.DETERMINISTIC
        self._hidden_size = hidden_size
        self.N_AGENTS = config.N_AGENTS
        self._hidden_size = hidden_size
        self._activation = activation

        self.global_cell = nn.GRU(hidden_size, self._deter_size)
        #self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1)  
        self.global_rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE * config.N_AGENTS + self.global_stoch_size)  
        self.global_stochastic_prior_model = DiscreteLatentDist(self.global_deter_size, config.GLOBAL_N_CATEGORICALS, config.GLOBAL_N_CLASSES, self._hidden_size)
        self.agent_cell = nn.GRU(hidden_size, self._deter_size)
        self.agent_rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE + self._stoch_size)       
        self.agent_stochastic_prior_model = DiscreteLatentDist(self._deter_size + self.global_stoch_size, config.N_CATEGORICALS, config.N_CLASSES, self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_joint_actions, prev_agent_states, prev_global_states, mask=None):
        if(mask is not None):
          print("Mask is not None")
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]
        global_stoch_input = self.global_rnn_input_model(torch.cat([prev_joint_actions, prev_global_states.stoch], dim=-1))
        global_deter_state = self.global_cell(global_stoch_input.reshape(1, batch_size * n_agents, -1),
                                 prev_global_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        global_logits, global_stoch_state = self.global_stochastic_prior_model(global_deter_state)
        agent_stoch_input = self.agent_rnn_input_model(torch.cat([prev_actions, prev_agent_states.stoch], dim=-1))
        agent_deter_state = self.agent_cell(agent_stoch_input.reshape(1, batch_size * n_agents, -1),
                                 prev_agent_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1) 
        agent_logits, agent_stoch_state = self.agent_stochastic_prior_model(torch.cat([agent_deter_state, global_stoch_state], dim = -1))

        return RSSMState(logits=agent_logits, stoch=agent_stoch_state, deter=agent_deter_state), RSSMState(logits=global_logits, stoch=global_stoch_state, deter=global_deter_state)


class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self.config = config
        self._transition_model = transition_model
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self.global_stoch_size = config.GLOBAL_STOCHASTIC
        self.global_deter_size = config.DETERMINISTIC
        self._hidden_size = config.HIDDEN
        self._activation = nn.ReLU    
        self.agent_stochastic_posterior_model = DiscreteLatentDist(self._deter_size + config.EMBED, config.N_CATEGORICALS,
                                                              config.N_CLASSES, config.HIDDEN)
        #self._attention_stack = AttentionEncoder(3, self._stoch_size, self._stoch_size, dropout=0.1)
        self.global_stochastic_posterior_model = DiscreteLatentDist(self.global_deter_size + self._stoch_size + config.STATE_EMBED, config.GLOBAL_N_CATEGORICALS,
                                                              config.GLOBAL_N_CLASSES, config.HIDDEN)

    def initial_agent_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def initial_global_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self.global_stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self.global_stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, obs_embed, state_embed, prev_actions, prev_joint_actions, prev_agent_states, prev_global_states, mask=None):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_agent_states, prior_global_states = self._transition_model(prev_actions, prev_joint_actions, prev_agent_states, prev_global_states, mask)
        x_agent = torch.cat([prior_agent_states.deter, obs_embed], dim=-1)
        agent_logits, agent_stoch_state = self.agent_stochastic_posterior_model(x_agent)
        x_global = torch.cat([prior_global_states.deter, state_embed, agent_stoch_state], dim=-1)
        global_logits, global_stoch_state = self.global_stochastic_posterior_model(x_global) 
        posterior_agent_states = RSSMState(logits=agent_logits, stoch=agent_stoch_state, deter=prior_agent_states.deter)
        posterior_global_states = RSSMState(logits=global_logits, stoch=global_stoch_state, deter=prior_global_states.deter)
        return prior_agent_states, posterior_agent_states, prior_global_states, posterior_global_states

def rollout_representation(config, model, steps, obs_embed, state_embed, action, joint_action, prev_agent_states, prev_global_states, done, test=False):
    """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
        :param action: size(time_steps, batch_size, n_agents, action_size)
        :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
        """
    agent_priors = []
    agent_posteriors = []
    global_priors = []
    global_posteriors = []
   
    for t in range(steps):

        prior_agent_states, posterior_agent_states, prior_global_states, posterior_global_states = model.representation(obs_embed[t], state_embed[t], \
                            action[t], joint_action[t], prev_agent_states, prev_global_states) #TODO GLOBAL POST AND PRI           

        prev_agent_states = posterior_agent_states.map(lambda x: x * (1.0 - done[t]))
        prev_global_states = posterior_global_states.map(lambda x: x * (1.0 - done[t])) 
        agent_priors.append(prior_agent_states)
        global_priors.append(prior_global_states)

        agent_posteriors.append(posterior_agent_states)
        global_posteriors.append(posterior_global_states)
    agent_prior = stack_states(agent_priors, dim=0)
    agent_post = stack_states(agent_posteriors, dim=0)

    global_prior = stack_states(global_priors, dim=0)
    global_post = stack_states(global_posteriors, dim=0)

    return agent_prior.map(lambda x: x[:-1]), agent_post.map(lambda x: x[:-1]), agent_post.deter[1:],\
                global_prior.map(lambda x: x[:-1]), global_post.map(lambda x: x[:-1]), global_post.deter[1:]

def rollout_policy(transition_model, av_action, steps, policy, prev_agent_state, prev_global_state):
    """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    agent_state = prev_agent_state
    global_state = prev_global_state
    agent_next_states = []
    global_next_states = []
    actions = []
    av_actions = []
    policies = []
    for t in range(steps):
        agent_feat = agent_state.get_features().detach()
        global_feat = global_state.get_features().detach()
        action, pi = policy(agent_feat)
        if av_action is not None:
            avail_actions = av_action(torch.cat([agent_feat, global_feat], dim = -1)).sample()
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(avail_actions.squeeze(0))
        agent_next_states.append(agent_state)
        global_next_states.append(global_state)
        policies.append(pi)
        actions.append(action)
        joint_action = action.clone().reshape(-1, action.shape[1]*action.shape[2])
        joint_action = torch.repeat_interleave(joint_action.unsqueeze(dim = 1), action.shape[1], dim = 1)
        agent_state, global_state = transition_model(action, joint_action, agent_state, global_state)
    return {"agent_imag_states": stack_states(agent_next_states, dim=0),
            "global_imag_states": stack_states(global_next_states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0)}
            