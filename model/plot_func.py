import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from init import get_image,get_des_file_rendering

# Plot the model performance
def plot_performance(rewards,level):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards\n during\n episode",rotation=0, labelpad=40)
    plt.title("Level "+level)
    plt.legend()
    plt.show()

# Visualize the playing process (live!)
def live_play_viz(env, agent):
    state = env.reset()
    n = 0
    done = False
    plt.figure(1)
    while not done:
        n += 1
        action = agent.take_action(state)
        state, reward, done, info = env.step(action)
        agent.update(state, reward)
        
        plt.imshow(get_image(state['pixel']))

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.01)
        plt.clf()

        display.display(plt.gcf())
        display.clear_output(wait=True)

def live_play_viz_es(env, agent):
    state = env.reset()
    n = 0
    done = False
    plt.figure(1)
    while not done:
        n += 1
        action = agent.take_action(state)
        state, reward, done, info = env.step(action)
        agent.update(reward)
        
        plt.imshow(get_image(state['pixel']))

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.01)
        plt.clf()

        display.display(plt.gcf())
        display.clear_output(wait=True)

# Plot the q value table
def extract_values(dictionary):
    x_values = [key[0][0] for key in dictionary.keys()]
    y_values = [key[0][1] for key in dictionary.keys()]
    z_values = list(dictionary.values())
    return x_values, y_values, z_values

def plot_SVfunction(agent,algoname):
    q_policy_values = agent.q

    x_list, y_list, z_list = extract_values(q_policy_values)

    fig, ax = plt.subplots(ncols=1, figsize=(5, 5), subplot_kw={'projection': '3d'})

    ax.plot_trisurf(x_list, y_list, z_list, cmap='Blues')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(algoname)

    plt.show()
