import sys
import argparse
import numpy as np

from game import SnakeGameAI

sys.path.append('../../..')
from machine_learning.agent import Agent

def play_game(game, agent, decay_rate):
    done = False

    while not done:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        processed_move = [0] * 3
        processed_move[np.argmax(final_move)] = 1

        # perform move and get new state
        reward, done, score = game.play_step(processed_move)
        state_new = game.get_state()

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

    agent.epsilon_decay(decay_rate)
    game.reset()

    return score

def train(model_path_name):
    agent = Agent([11, 512, 3], batch_size=1000)
    game = SnakeGameAI()

    decay_rate = 0.99

    n_games = 500

    try:
        for current_game in range(n_games):
            score = play_game(game, agent, decay_rate)

            # train agent with replay buffer
            cost = agent.train_long_memory()

            print('Game', current_game, 'Score', score, 'Cost:', cost)
    except KeyboardInterrupt:
        pass

    print('Saving model...')
    agent.save(model_path_name)
    
def evauluate(model_filename):
    agent = Agent([11, 40, 40, 3])
    agent.load(model_filename)
    agent.epsilon_decay(0.0)

    game = SnakeGameAI()

    n_games = 200
    for current_game in range(n_games):

        # train agent with replay buffer
        score = play_game(game, agent, 0.0)

        print('Game', current_game, 'Score', score)

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