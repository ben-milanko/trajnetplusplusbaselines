from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from random import random

count = 0

def fyp_visualise(scene):

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    cmap = plt.get_cmap('turbo')

    tracks = []
    track_plots = []

    frames = scene[2]
    # print(scene)

    

    for track in frames:
        col = random()
        x_data = [track[0].x]
        y_data = [track[0].y]
        track_plot, = plt.plot(x_data, y_data, '-',  color=cmap(col))

        tracks.append([x_data, y_data])
        track_plots.append(track_plot)

    def update(_):
        global count
        # print(count)

        for i, track in enumerate(frames):
            # print(track)
            x_data = tracks[i][0]
            y_data = tracks[i][1]
            # print(len(track), count)
            if len(track) > count:
                data = track[count]
                x_data.append(data.x)
                y_data.append(data.y)

            tracks[i] = [x_data, y_data]

            track_plots[i].set_data(x_data, y_data)
        
        count += 1
        return track_plots

    ani = FuncAnimation(fig, update, blit=True)
    Writer = writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=18000)
    ani.save('im.mp4', writer=writer)

    plt.show()
