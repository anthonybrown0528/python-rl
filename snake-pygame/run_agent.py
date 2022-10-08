from agent import Agent
from game import SnakeGameAI
import helper

import numpy as np

import sys

def execute(model_filename):
    agent = Agent()
    game = SnakeGameAI()

    agent.model.load(model_filename)
    agent.epsilon = 0

    scores = []
    mean_scores = []

    record = 0
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        _, done, score = game.play_step(final_move)
        agent.get_state(game)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1

            scores.append(score)
            scores_np = np.array(scores)

            mean_scores.append(np.average(scores_np))
            mean_scores_np = np.array(mean_scores)

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            helper.plot([scores_np, mean_scores_np])

if __name__ == '__main__':
    model_path = 'tmp.pth'

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    
    print(f'{model_path} loaded as model')
    execute(model_path)