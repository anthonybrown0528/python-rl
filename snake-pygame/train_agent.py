import sys

from machine_learning.agent import Agent
from game import SnakeGameAI

def train(model_path_name):
    plot_scores = []
    plot_mean_scores = []
    plot_cost = []

    total_score = 0
    record = 0

    agent = Agent([11, 1024, 3])
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()

            agent.n_games += 1
            agent.network.epsilon -= 1

            cost = agent.train_long_memory()
            plot_cost.append(cost)

            if score > record:
                record = score
                agent.network.save(model_path_name)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            
if __name__ == '__main__':
    model_path = 'tmp.pth'

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    
    print(f'Model will be saved as {model_path}')
    train(model_path)