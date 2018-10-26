from argparse import Namespace
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2


def namespace_to_dict(namespace):
    """Deep (recursive) transform from Namespace to dict"""
    dct = dict()
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def read_cfg(config_file):
    """ Parse yaml type config file. """
    with open(config_file) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    cfg = dict_to_namespace(config_data)
    return cfg


def dict_to_namespace(dct):
    """Deep (recursive) transform from Namespace to dict"""
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def processFrame(frame):
    '''
    Process the input frame as described in the Nature Paper
    Take the maximum pixel color for two conesecutive images
    Then rescale the image to 84 * 84 and conver it to grayScale
    Returns an array of numpy uint8 84 * 84 * 1
    '''

    processed_frame = frame.astype(np.float32)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(
        processed_frame, (84, 84), interpolation=cv2.INTER_AREA)

    x_t = np.reshape(resized_frame, [84, 84, 1])
    return x_t.astype(np.uint8)


def statistics(episode, steps_done, nr_updates, reward_per_episode, eps, eps_steps):

    print("Episode number {}: reward {}, steps_done {}, nr_updates {}, eps {}, steps_per_episode {}".
          format(episode, reward_per_episode, steps_done, nr_updates, eps, eps_steps))


def plot_rewardEvaluation(cfg, agent, nr_updates, q_network, logger):

    nr_episodes = 10

    if nr_updates % cfg.epoch_size == 0:
        q_network.train(False)
        current_rewardEstimate, ep_scores = agent.evalAgent(
            nr_episodes, q_network)
        q_network.train(True)
        agent.reward_history.append(current_rewardEstimate)

        logger.write("Save plot" + str(nr_updates) + "\n")
        logger.flush()
        plt.plot(agent.reward_history)
        plt.xlabel('Nr Epochs')
        plt.ylabel('Mean Score per episode')
        plt.title('After' + str(nr_updates))
        plt.savefig('History_Rewards2')
