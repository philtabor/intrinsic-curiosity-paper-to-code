import torch.multiprocessing as mp
from actor_critic import ActorCritic
from icm import ICM
from shared_adam import SharedAdam
from worker import worker


class ParallelEnv:
    def __init__(self, env_id, global_idx,
                 input_shape, n_actions, num_threads, icm=False):
        names = [str(i) for i in range(num_threads)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        if icm:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters(), lr=1e-4)
        else:
            global_icm = None
            global_icm_optim = None

        self.ps = [mp.Process(target=worker,
                              args=(name, input_shape, n_actions,
                                    global_actor_critic, global_optim, env_id,
                                    num_threads, global_idx, global_icm,
                                    global_icm_optim, icm))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
