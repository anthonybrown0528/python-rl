import sys
import argparse
import numpy as np

from game import SnakeGameAI

sys.path.append('../../..')
from machine_learning.agent import Agent

def train(model_path_name):
    plot_scores = []
    plot_mean_scores = []
    plot_cost = []

    total_score = 0
    record = 0

    agent = Agent([11, 1024, 3])
    game = SnakeGameAI()

    n_games = 0

    decay_rate = 0.99


    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()

            n_games += 1
            agent.epsilon_decay(decay_rate)

            cost = agent.train_long_memory()
            plot_cost.append(cost)

            if score > record:
                record = score
                agent.network.save(model_path_name)

            print('Game', n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)

def evauluate(model_filename):
    agent = Agent([11, 1024, 3], eval=True)
    game = SnakeGameAI()

    agent.network.load(model_filename)

    scores = []
    mean_scores = []

    record = 0
    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        _, done, score = game.play_step(final_move)

        if done:
            # train long memory, plot results
            game.reset()
            n_games += 1

            scores.append(score)
            scores_np = np.array(scores)

            mean_scores.append(np.average(scores_np))

            if score > record:
                record = score

            print('Game', n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':

    # Parse CLI arguments

    parser = argparse.ArgumentParser(description='Trains/evaluates a learning model playing the Snake game')

    parser.add_argument('-f', '--filename', default='tmp.pth', help='An optional argument for specifying file name of the model to store or load')
    parser.add_argument('-e', '--eval', action='store_true', help='An optional argument for specifying the program to run in evaluation mode instead of training mode')

    args = parser.parse_args()

    # Run program

    model_path = args.filename
    
    if args.eval:
        print(f'{model_path} loaded as model')
        evauluate(model_path)
    else:
        print(f'Model will be saved as {model_path}')
        train(model_path)