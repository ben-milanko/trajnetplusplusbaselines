import numpy as np
from .fyp_visualise import fyp_visualise
from .action import ActionXY
from .model_predictive_rl import ModelPredictiveRL
from typing import Iterable
from .state import FullState, JointState, ObservableState
from .visualise import visualise
import torch
from tqdm import tqdm
import logging

FRAMERATE = 3
PED_SIZE = 0.5

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

        self.reward_sum = None
        self.no_move_ade = None
        self.collisions = None
        self.collision_rate = None

    def run_k_episodes(self, k, policy: ModelPredictiveRL, phase='train', clip_scene=4, label='', supervised=False, show_vis=True, episode=0, reward='position'):
        print("")
        pbar = tqdm(total=k)
        pbar.set_description(label)

        robot_positions = []
        states: Iterable[JointState] = []
        actions: Iterable[ActionXY] = []
        rewards: Iterable[float] = []
        samples = 0
        self.reward_sum = 0
        self.no_move_ade = 0
        self.collisions = 0

        while samples < k:
            if phase == 'train':
                if self.train_scene_iter >= len(self.train_scenes):
                    self.train_scene_iter = 0
                scene = self.train_scenes[self.train_scene_iter]
            else:
                if self.val_scene_iter >= len(self.val_scenes):
                    self.val_scene_iter = 0
                scene = self.val_scenes[self.val_scene_iter]

            # fyp_visualise(scene)
            if supervised:
                states, actions, rewards, robot_positions, goal = self.supervised_explorer(
                    scene, policy=policy, phase='train', clip_scene=4, early_quit=k-samples, pbar=pbar)
            else:
                states, actions, rewards, robot_positions, goal = self.rl_explorer(
                    scene, policy=policy, phase='train', clip_scene=4, early_quit=k-samples, pbar=pbar, reward=reward)

            if show_vis:
                visualise(states, robot_positions, goal, rewards, episode=episode,
                          scene=self.train_scene_iter if phase == 'train' else self.val_scene_iter)

            samples += len(states)

            self.update_memory(states, actions, rewards)
            self.reward_sum += sum(rewards)

            states = []
            actions = []
            rewards = []

            if phase == 'train':
                self.train_scene_iter += 1
            else:
                self.val_scene_iter += 1

        pbar.close()

        self.collision_rate = self.collisions/k if self.collisions is not None else None

        logging.info(
            f'Episode total reward: {self.reward_sum:.2f}, ADE: {-(self.reward_sum/k):.2f}, No Move ADE: {self.no_move_ade/k:.2f}{f", Collision rate: {self.collisions/k:.2f}" if self.collisions is not None else ""}')

        return states, actions, self.reward_sum

    def supervised_explorer(self, scene, policy: ModelPredictiveRL, phase='train', clip_scene=1, early_quit=None, pbar=None):
        robot_positions = []
        states: Iterable[JointState] = []
        actions: Iterable[ActionXY] = []
        rewards: Iterable[float] = []

        samples = 1

        frames = scene[2]

        if len(frames) == 1:
            if phase == 'train':
                self.train_scene_iter += 1
            else:
                self.val_scene_iter += 1
            return

        primary = frames[0]
        # Exclude the final frame of the scene as reward must be calculated against the next frame
        scene_length = len(primary)-1

        for i in range(1, scene_length):
            vx = (primary[i].x - primary[i-1].x)/(1/FRAMERATE)
            vy = (primary[i].y - primary[i-1].y)/(1/FRAMERATE)
            robot_state = FullState(
                vx, vy, primary[-1].x - primary[i].x, primary[-1].y - primary[i].y)

            human_states = []

            # Reduce the number of pedestrians
            # Can only use the next clip scene pedestrians from the scene after primary
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
                        frame.x - primary[i].x, frame.y - primary[i].y, vx, vy))
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

            euclidian_dist = np.sqrt(np.square(
                action.vx - vx_next)+np.square(action.vy-vy_next))

            self.no_move_ade += np.sqrt(np.square(vx_next) +
                                        np.square(vy_next))

            reward = -euclidian_dist  # 1/(1+euclidian_dist)

            rewards.append(reward)

            # Update progress graphs
            pbar.update(1)
            if early_quit <= samples:
                return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)
            samples += 1

        return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)

    def rl_explorer(self, scene, policy: ModelPredictiveRL, phase='train', clip_scene=4, early_quit=None, pbar=None, reward='velocity_next_gt'):

        robot_positions = []
        states: Iterable[JointState] = []
        actions: Iterable[ActionXY] = []
        rewards: Iterable[float] = []

        samples = 1
        collision = False

        frames = scene[2]

        primary = frames[0]

        if len(frames) == 1:
            if phase == 'train':
                self.train_scene_iter += 1
            else:
                self.val_scene_iter += 1
            return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)

        scene_length = len(primary)-1

        robot_pos = (primary[0].x, primary[0].y)
        robot_positions.append([robot_pos, robot_pos])

        for i in range(1, scene_length):
            vx = (robot_pos[0] - primary[i-1].x)/(1/FRAMERATE)
            vy = (robot_pos[1] - primary[i-1].y)/(1/FRAMERATE)
            robot_state = FullState(
                vx, vy, primary[-1].x - robot_pos[0], primary[-1].y - robot_pos[1])

            human_states = []
            
            # closest = []
            # for j in range(1, len(frames)):
            #     frame = frames[j][i]
            #     if len(closest) == 0:
            # Reduce the number of pedestrians
            # Can only use the next clip scene pedestrians from the scene after primary
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

                    if np.linalg.norm((frame.x - robot_pos[0], frame.y - robot_pos[1])) < PED_SIZE*2:
                        collision = True
                        self.collisions += 1
                    human_states.append(ObservableState(
                        frame.x - robot_pos[0], frame.y - robot_pos[1], vx, vy))
                else:
                    human_states.append(human_states[-1])

            state = JointState(robot_state, human_states)

            # Predict an action based on the state
            action = policy.predict(state)

            # Move the agent as per the predicted action
            robot_pos = (robot_pos[0]+action.vx*(1/FRAMERATE),
                         robot_pos[1]+action.vy*(1/FRAMERATE))
            robot_positions.append([robot_pos, (primary[i].x, primary[i].y)])

            actions.append(action)
            states.append(state.to_tensor(
                device=self.device))

            # Reward is 1 over 1 plus the euclidian distance between the predicted action and the actual position
            if collision:
                rewards.append(-100)
                return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)

            if reward == 'velocity':
                vx_next = (primary[i+1].x - primary[i].x)/(1/FRAMERATE)
                vy_next = (primary[i+1].y - primary[i].y)/(1/FRAMERATE)

                euclidian_dist = np.linalg.norm((
                    action.vx - vx_next, action.vy-vy_next))

                self.no_move_ade += np.linalg.norm((vx_next, vy_next))
            elif reward == 'position':
                euclidian_dist = np.linalg.norm(
                    (primary[i+1].x-robot_pos[0], primary[i+1].y-robot_pos[1]))

                self.no_move_ade += np.linalg.norm(
                    (primary[i+1].x-primary[0].x, primary[i+1].y-primary[0].y))
            elif reward == 'next_ground_truth':

                mag = np.linalg.norm((primary[i+1].x - robot_pos[0], primary[i+1].y - robot_pos[1]))
                vx_next = ((primary[i+1].x - robot_pos[0])/mag)*1.5/(1/FRAMERATE)
                vy_next = ((primary[i+1].y - robot_pos[1])/mag)*1.5/(1/FRAMERATE)

                euclidian_dist = np.linalg.norm((
                    action.vx - vx_next, action.vy-vy_next))

                self.no_move_ade += np.linalg.norm((vx_next, vy_next))

            reward = -euclidian_dist  # 1/(1+euclidian_dist)

            rewards.append(reward)

            # Update progress graphs
            pbar.update(1)
            if early_quit <= samples:
                return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)
            samples += 1

        return states, actions, rewards, robot_positions, (primary[-1].x, primary[-1].y)

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
        reward = self.reward_sum
        collision_rate = self.collision_rate

        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        if collision_rate is not None:
            self.writer.add_scalar(tag_prefix + '/collision_rate', collision_rate, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
