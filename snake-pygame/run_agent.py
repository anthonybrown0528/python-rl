from agent import Agent
from game import SnakeGameAI

import sys

def execute(model_filename):
    agent = Agent()
    game = SnakeGameAI()

    agent.model.load(model_filename)
    agent.epsilon = 0

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

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2
    except AssertionError:
        print('Please enter a filename to locate existing model!')
        exit(-1)

    execute(sys.argv[1])