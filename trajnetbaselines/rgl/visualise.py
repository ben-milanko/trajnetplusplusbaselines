from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualise(states: Iterable, robot_pos: Iterable[tuple], goal, rewards, episode=0, scene=0):

    frames = []
    for i, state in enumerate(states):
        frame = [robot_pos[i]] + [rewards[i]] + [[float(human[0]), float(human[1])] for human in state[1]]
        frames.append(frame)
    
    fig, ax = plt.subplots()
    humans, = plt.plot([], [], 'ro')
    robot, = plt.plot([], [], 'bo')
    ground_truth, = plt.plot([], [], 'ko')
    goal, = plt.plot([goal[0]], [goal[1]], 'go')
    text = plt.text(-18, 18, '', fontsize=10)

    humans.set_label('Pedestrian')
    robot.set_label('Primary')
    ground_truth.set_label('Ground Truth')
    goal.set_label('Goal')

    def init():
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_title(f'Episode: {episode}, Scene: {scene}')
        ax.legend()
        
        return humans,robot,ground_truth

    def update(frame: Iterable):
        human_pos = frame[2:-1]
        xdata = [state[0] for state in human_pos]
        ydata = [state[1] for state in human_pos]
        
        humans.set_data(xdata, ydata)
        robot.set_data(frame[0][0][0], frame[0][0][1])
        ground_truth.set_data(frame[0][1][0], frame[0][1][1])

        text.set_text(f'Reward: {frame[1]:.2f}')

        return humans,robot,ground_truth,text

    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True)
    plt.show()