import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy


def model_loss(config, model, obs, global_state, action, av_action, reward, done, fake, last):

    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    state = global_state.reshape(time_steps, batch_size, -1).unsqueeze(dim = -2).expand(-1, -1, n_agents, -1)
    joint_action = action.reshape(-1, batch_size, n_agents*action.shape[-1]).unsqueeze(dim = 2).expand(-1, -1, n_agents, -1)
    #joint_action = torch.repeat_interleave(joint_action.unsqueeze(dim = 2), n_agents, dim = 2)

    obs_embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))      
    obs_embed = obs_embed.reshape(time_steps, batch_size, n_agents, -1)
    state_embed = model.state_encoder(state.reshape(-1, n_agents, state.shape[-1]))

    #state_embed = model.attention_stack(obs_embed.reshape(-1, n_agents, obs_embed.shape[-1])).reshape(time_steps, batch_size,-1)
    state_embed = state_embed.reshape(time_steps, batch_size, n_agents, -1)
    prev_agent_state = model.representation.initial_agent_state(batch_size, n_agents, device=obs.device)
    prev_global_state = model.representation.initial_global_state(batch_size, n_agents, device=obs.device)

    agent_prior, agent_post, agent_deters, global_prior, global_post, global_deters = rollout_representation(config, model, time_steps, obs_embed, state_embed, action, \
                                                                                                              joint_action, prev_agent_state, prev_global_state, last)       
    agent_feat = torch.cat([agent_post.stoch, agent_deters], -1) 
    global_feat = torch.cat([global_post.stoch, global_deters], -1)      
    feat = torch.cat([agent_feat, global_feat], dim = -1)
    agent_feat_dec = agent_post.get_features()
    global_feat_dec = global_post.get_features()
    feat_dec = torch.cat([agent_post.get_features(), global_post.get_features()], dim = -1)

    agent_reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           agent_feat_dec.reshape(-1, n_agents, agent_feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))   

    # g_recon_loss, _ = rec_loss(model.state_decoder,
    #                                        global_feat_dec.reshape(-1, n_agents, global_feat_dec.shape[-1]),
    #                                        state[:-1].reshape(-1, n_agents, state.shape[-1]),
    #                                        1. - fake[:-1].reshape(-1, n_agents, 1)) 

    #g_reward_loss = F.smooth_l1_loss(model.g_reward_model(global_feat), reward[1:]) 
    reward_loss = F.smooth_l1_loss(model.reward_model(agent_feat), reward[1:]) 
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))     
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1) 
    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1)) 
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.

    agent_div = state_divergence_loss(agent_prior, agent_post,  config.N_CATEGORICALS, config.N_CLASSES)
    global_div = state_divergence_loss(global_prior, global_post, config.GLOBAL_N_CATEGORICALS, config.GLOBAL_N_CLASSES)

    model_loss = agent_div + global_div + reward_loss + dis_loss + agent_reconstruction_loss + pcont_loss + av_action_loss# + g_reward_loss
    
    if np.random.randint(20) == 4:
        wandb.log({'Model/reward_loss': reward_loss, 'Model/agent_div': agent_div, 'Model/global_div': global_div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': agent_reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss

def actor_rollout(obs, global_state, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    with FreezeParameters([model]):
        state = global_state.reshape(time_steps, batch_size, -1).unsqueeze(dim = -2).expand(-1, -1, n_agents, -1)
        joint_action = action.reshape(-1, obs.shape[1], n_agents*action.shape[-1]).unsqueeze(dim = 2).expand(-1, -1, n_agents, -1)
        #joint_action = torch.repeat_interleave(joint_action.unsqueeze(dim = 2), n_agents, dim = 2)

        obs_embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        obs_embed = obs_embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        state_embed = model.state_encoder(state.reshape(-1, n_agents, state.shape[-1]))
        #state_embed = model.attention_stack(obs_embed.reshape(-1, n_agents, obs_embed.shape[-1])).reshape(time_steps, batch_size,-1)
        state_embed = state_embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_agent_state = model.representation.initial_agent_state(obs.shape[1], obs.shape[2], device=obs.device)
        prev_global_state = model.representation.initial_global_state(obs.shape[1], obs.shape[2], device=obs.device)
        agent_prior, agent_post, _, global_prior, global_post, _ = rollout_representation(config, model, obs.shape[0], obs_embed, state_embed, action, \
                                                                    joint_action, prev_agent_state, prev_global_state, last)       

        agent_post = agent_post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        global_post = global_post.map(lambda x: x.reshape((state.shape[0] - 1) * state.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, agent_post, global_post)
    agent_imag_feat = items["agent_imag_states"].get_features()
    global_imag_feat = items["global_imag_states"].get_features()
    a_imag_rew_feat = torch.cat([items["agent_imag_states"].stoch[:-1], items["agent_imag_states"].deter[1:]], -1)
    g_imag_rew_feat = torch.cat([items["global_imag_states"].stoch[:-1], items["global_imag_states"].deter[1:]], -1)
    imag_rew_feat = torch.cat([a_imag_rew_feat, g_imag_rew_feat], dim = -1)
    returns = critic_rollout(model, critic, agent_imag_feat, global_imag_feat, imag_rew_feat, items["actions"], 
                             items["agent_imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])),\
                             items["global_imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), agent_imag_feat[:-1].detach(), global_imag_feat[:-1].detach(), returns.detach()] #MIGHT NEED TO DO REPEAT ON GLOBALS
    return [batch_multi_agent(v, n_agents) for v in output]

def critic_rollout(model, critic, agent_states, global_states, rew_states, actions, agent_raw_states, global_raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, agent_raw_states, global_raw_states) #TODO
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(agent_states, global_states, actions) 
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA) 
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, agent_states, global_states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    joint_actions = actions.reshape(-1, actions.shape[-2] * actions.shape[-1])
    joint_actions = torch.repeat_interleave(joint_actions.unsqueeze(dim = -2), actions.shape[-2], dim = -2)
    agent_next_state, global_next_state = model.transition(actions, joint_actions, agent_states, global_states)
    imag_rew_feat = torch.cat([agent_states.stoch, agent_next_state.deter], -1)
    #g_imag_rew_feat = torch.cat([global_states.stoch, global_next_state.deter], -1)
    #imag_rew_feat = torch.cat([a_imag_rew_feat, g_imag_rew_feat], dim = -1)
    return calculate_reward(model, imag_rew_feat)


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight):
    _, new_policy = actor(imag_states)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, actions, agent_imag_feat, global_imag_feat, targets):
    value_pred = critic(agent_imag_feat, global_imag_feat, actions)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
