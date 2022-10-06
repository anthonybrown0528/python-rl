import os
import sys
from agent import Agent
from game import SnakeGameAI
from helper import plot

def train(model_path_name):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()

            agent.n_games += 1
            agent.epsilon -= 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(model_path_name)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2
    except AssertionError:
        print('Please enter a filename to store trained model!')
        exit(-1)

    if os.path.exists(os.path.join('model', sys.argv[1])):
        raise RuntimeError('A file with the name: ' + sys.argv[1] + ' already exists under the /model/ folder!')

    train(sys.argv[1])