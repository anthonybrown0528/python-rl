from machine_learning.agent import Agent
from game import SnakeGameAI

import numpy as np

import sys

def execute(model_filename):
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
            agent.n_games += 1

            scores.append(score)
            scores_np = np.array(scores)

            mean_scores.append(np.average(scores_np))

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    model_path = 'tmp.pth'

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    
    print(f'{model_path} loaded as model')
    execute(model_path)