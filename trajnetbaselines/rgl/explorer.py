import numpy
from .action import ActionXY
from .model_predictive_rl import ModelPredictiveRL
from typing import Iterable
from .state import FullState, JointState, ObservableState
import torch
from tqdm import tqdm

FRAMERATE = 3


class Explorer(object):
    def __init__(self, device, train_scenes, val_scenes, memory=None, gamma=None, target_policy=None, writer=None):
        self.statistics = None
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.writer = writer

        self.train_scenes = train_scenes
        self.val_scenes = val_scenes
        self.train_scene_iter = 0
        self.val_scene_iter = 0

        self.statistics = None

    def run_k_episodes(self, k, policy: ModelPredictiveRL, phase='train', clip_scene=4, label=''):
        print("")
        pbar = tqdm(total=k)
        pbar.set_description(label)

        states: Iterable[JointState] = []
        actions: Iterable[ActionXY] = []
        rewards: Iterable[float] = []
        samples = 0
        self.statistics = 0

        while samples < k:
            if phase == 'train':
                scene = self.train_scenes[self.train_scene_iter]
            else:
                scene = self.val_scenes[self.val_scene_iter]

            frames = scene[2]

            if len(frames) == 1:
                if phase == 'train':
                    self.train_scene_iter += 1
                else:
                    self.val_scene_iter += 1
                continue

            primary = frames[0]
            # Exclude the final frame of the scene as reward must be calculated against the next frame
            scene_length = len(primary)-1

            for i in range(scene_length):
                if i == 0:
                    vx, vy = 0, 0
                else:
                    vx = (primary[i].x - primary[i-1].x)/(1/FRAMERATE)
                    vy = (primary[i].y - primary[i-1].y)/(1/FRAMERATE)
                robot_state = FullState(
                    primary[i].x, primary[i].y, vx, vy, primary[-1].x, primary[-1].y)

                human_states = []
                # Can only use the next 4 pedestrians from the scene after primary
                for j in range(1, clip_scene+1):
                    if len(frames) > j:
                        if len(frames[j]) <= i:
                            frame = frames[j][len(frames[j])-1]
                            vx, vy = 0, 0
                        else:
                            frame = frames[j][i]

                        # Velocity should be 0 at the start or end of the run
                        if i == 0 or len(frames[j]) < i:
                            vx, vy = 0, 0
                        else:
                            vx = (frame.x - frames[j][i-1].x)/(1/FRAMERATE)
                            vy = (frame.y - frames[j][i-1].y)/(1/FRAMERATE)

                        human_states.append(ObservableState(
                            frame.x, frame.y, vx, vy))
                    else:
                        human_states.append(human_states[-1])

                state = JointState(robot_state, human_states)

                # Predict an action based on the state
                action = policy.predict(state)

                actions.append(action)
                states.append(state.to_tensor(
                    device=self.device))

                # Reward is 1 over 1 plus the euclidian distance between the predicted action and the actual position

                vx_next = (primary[i+1].x - primary[i].x)/(1/FRAMERATE)
                vy_next = (primary[i+1].y - primary[i].y)/(1/FRAMERATE)

                # print(vx_next, vy_next)
                # print(action.vx, action.vy)
                
                euclidian_dist = numpy.sqrt(numpy.square(
                    action.vx - vx_next)+numpy.square(action.vy-vy_next))

                reward = 1/(1+euclidian_dist)
                
                # print(euclidian_dist)
                # input()
                rewards.append(reward)

                # Update progress graphs
                pbar.update(1)
                samples += 1
                if len(states) == k:
                    break

            self.update_memory(states, actions, rewards)
            self.statistics += sum(rewards)

            states = []
            actions = []
            rewards = []

            if phase == 'train':
                self.train_scene_iter += 1
            else:
                self.val_scene_iter += 1

        pbar.close()

        return states, actions, sum(rewards)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push(
                    (state[0], state[1], value, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        # sr, cr, time, reward, avg_return = self.statistics
        reward = self.statistics
        # self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        # self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        # self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        # self.writer.add_scalar(tag_prefix + '/avg_return',
        #                        avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
