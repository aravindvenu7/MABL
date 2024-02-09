#import ray
import wandb
#wandb.login(key = [85a593faec6432f3c0804364fdd9006072f160c1])
from agent.workers.DreamerWorker import DreamerWorker
import numpy as np, random
import torch
import matplotlib.pyplot as plt
def set_all_seeds(seed):
  random.seed(seed)
  #os.environ('PYTHONHASHSEED') = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def plot_rewards(rewards, q, label):

    avg_rew = []
    j = 0
    
    while(j < len(rewards) - q):

      x = rewards[j:j+q]
      sum1 = np.sum(np.array(x)) /q
      avg_rew.append(sum1)
      j = j+1

    plt.plot(avg_rew, label = label)

# class DreamerServer:
#     def __init__(self, n_workers, env_config, controller_config, model):
#         ray.init()

#         self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
#         self.tasks = [worker.run.remote(model) for worker in self.workers]

#     def append(self, idx, update):
#         self.tasks.append(self.workers[idx].run.remote(update))

#     def run(self):
#         done_id, tasks = ray.wait(self.tasks)
#         self.tasks = tasks
#         recvs = ray.get(done_id)[0]
#         return recvs


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        #set_all_seeds(100)
        self.learner = learner_config.create_learner()
        self.worker = DreamerWorker(1, env_config, controller_config)
        #self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        stats = []
        while True:
            rollout, info = self.worker.run(self.learner.model, self.learner.actor)
            self.learner.step(rollout)
            cur_steps += info["steps_done"]
            cur_episode += 1
            wandb.log({'reward': info["reward"], 'steps': cur_steps})
            stats.append(info["reward"])
            if(len(stats)%1==0):
              np.save('mamba_rew', np.array(stats))
            plot_rewards(stats, 2, 'Reward')
            plt.legend()
            plt.xlabel('Episodes')
            plt.ylabel('Episode_Rewards')
            plt.savefig('mamba.png')
            plt.close()          
            #print(cur_episode, self.learner.total_samples, info["reward"])
            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            if(cur_episode%100==0):
               self.worker.env.close()#self.server.append(info['idx'], self.learner.params())

