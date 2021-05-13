import argparse
from datetime import datetime

from .model_predictive_rl import ModelPredictiveRL
from .memory import ReplayMemory
from .data_load_utils import prepare_data
from .explorer import Explorer
from .action import ActionXY
from .mprl_trainer import MPRLTrainer

import logging
import os
import random
import sys
import torch
import socket
from pprint import pformat
import pprint
import importlib.util

from .. import __version__ as VERSION

import logging
import torch
from tensorboardX import SummaryWriter

pp = pprint.PrettyPrinter(indent=4)


def main(epochs=25):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--step_size', default=10, type=int,
                        help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')
    parser.add_argument('--loss', default='pred', choices=('L2', 'pred'),
                        help='loss objective, L2 loss (L2) and Gaussian loss (pred)')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),
                        help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    parser.add_argument('--visualise', action='store_true',
                        help='flag to visualise the validation stage')
    parser.add_argument('--reward', default='position',
                        choices=('position', 'velocity', 'next_ground_truth'),
                        help='type of reward')
    parser.add_argument('--supervised-trainer', action='store_true',
                        help='User supervised trainer rather than rl trainer')
                    
    # Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    # pretrain.add_argument('--load-full-state', default=None,
    #                       help='load a pickled full state dictionary before training')
    # pretrain.add_argument('--nonstrict-load-state', default=None,
    #                       help='load a pickled state dictionary before training')

    # Augmentations
    # parser.add_argument('--augment', action='store_true',
    #                     help='perform rotation augmentation')
    # parser.add_argument('--normalize_scene', action='store_true',
    #                     help='rotate scene so primary pedestrian moves northwards at end of observation')
    # parser.add_argument('--augment_noise', action='store_true',
    #                     help='flag to add noise to observations for robustness')
    # parser.add_argument('--obs_dropout', action='store_true',
    #                     help='perform observation length dropout')

    args = parser.parse_args()

    # Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    # Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.goals:
        args.output = 'OUTPUT_BLOCK/{}/rgl_goals_{}_{}.pkl'.format(
            args.path, args.type, args.output)
    else:
        args.output = 'OUTPUT_BLOCK/{}/rgl_{}_{}.pkl'.format(
            args.path, args.type, args.output)

    # configure logging
    from pythonjsonlogger import jsonlogger
    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter(
        '%(message)s %(levelname)s %(name)s %(asctime)s'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[
                        stdout_handler, file_handler])
    logging.info(pformat({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    }))

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args.path
    # Prepare data
    train_scenes, train_goals, _ = prepare_data(
        args.path, subset='/train/', sample=args.sample, goals=args.goals)
    val_scenes, val_goals, val_flag = prepare_data(
        args.path, subset='/val/', sample=args.sample, goals=args.goals)

    HUMAN_NUM = 4

    spec = importlib.util.spec_from_file_location(
        'config', 'trajnetbaselines/rgl/config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    policy_config = config.PolicyConfig()
    policy = ModelPredictiveRL()
    policy.configure(policy_config)
    policy.set_device(args.device)
    policy.set_phase('train')
    policy.set_time_step(1/3)

    train_config = config.TrainConfig(False)
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay

    # trainer_config
    if args.load_state is not None:
        policy.load_model(args.load_state)
    
    model = policy.get_model()

    memory = ReplayMemory(10000)

    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")

    writer = SummaryWriter(log_dir=f'OUTPUT_BLOCK/rgl_log/{date_time}')
    explorer = Explorer(args.device, train_scenes, val_scenes,  memory=memory,
                        gamma=policy.gamma, target_policy=policy, writer=writer)
    batch_size = 100
    optimizer = 'Adam'
    reduce_sp_update_frequency = False
    freeze_state_predictor = False
    detach_state_predictor = True
    share_graph_model = True

    trainer = MPRLTrainer(model, policy.state_predictor, memory, args.device, policy, writer, batch_size, optimizer, HUMAN_NUM,
                          reduce_sp_update_frequency=reduce_sp_update_frequency,
                          freeze_state_predictor=freeze_state_predictor,
                          detach_state_predictor=detach_state_predictor,
                          share_graph_model=share_graph_model)

    rl_learning_rate = train_config.train.rl_learning_rate
    trainer.set_learning_rate(rl_learning_rate)
    trainer.update_target_model(model)

    save_interval = 10
    target_update_interval = 10
    evaluation_interval = 10
    val_samples = 100

    episode = 0
    while episode < 10000:

        # Decay epsilon over time
        if episode < epsilon_decay:
            epsilon = epsilon_start + \
                (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        policy.set_epsilon(epsilon)

        # Collect k samples
        explorer.run_k_episodes(
            batch_size, policy, clip_scene=4, label=f"Training | Episode {episode}", episode=episode, show_vis=args.visualise, reward=args.reward, supervised=args.supervised_trainer)

        trainer.optimize_batch(batch_size, episode)
        explorer.log(f"RGL_Trajnet_Train", episode)

        episode += 1

        if episode % save_interval == 0:
            output_dir = f'OUTPUT_BLOCK/rgl_models/{args.reward}_{date_time}'
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            file_name = f'{output_dir}/rgl.episode{episode}.pth'
            file = open(file_name, 'wb')
            print(f'Saving model to {file_name}')
            policy.save_model(file)

        if episode % target_update_interval == 0:
            logging.info("Updating Model")
            trainer.update_target_model(model)

        if episode % evaluation_interval == 0:
            policy.set_phase('val')
            explorer.run_k_episodes(
                val_samples, policy, phase='val', clip_scene=4, label=f"Validation | Episode {episode}", show_vis=args.visualise, episode=episode)
            logging.info(
                f"Reward per sample: {(explorer.reward_sum/val_samples):.2f}, total reward: {explorer.reward_sum:.2f}")
            explorer.log(f"RGL_Trajnet_Val", episode)

            policy.set_phase('train')


if __name__ == '__main__':
    main()
