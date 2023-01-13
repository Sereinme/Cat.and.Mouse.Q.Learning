# Programming Homework 3 for
# Machine Learning, 2022 Autumn

import time

from qlearn import QLearning
from visual import Board


def run():
    """
    Run cat and mouse reinforcement learning
    """

    total_episode = 100
    bar_length = 50
    start = time.perf_counter()

    for episode in range(total_episode):

        # steps
        steps = 0

        # reset title
        env.set_title("Cat and Mouse [Episode = " + str(episode) + "]")

        # reset environment
        s = env.reset()

        while True:

            steps += 1

            # fresh environment
            env.render()

            # RL choose action using QLearning
            action = rl.choose_action(s)

            # take action and get environment feedback
            s_, reward, done, won = env.step(action)

            # RL update Q table
            rl.learn(s, action, reward, s_)

            # swap state
            s = s_

            # break loop while end of episode
            if done:
                print("\r" + " " * (bar_length + 20), end="")
                print("\rEpisode = " + str(episode), end=", ")
                if won:
                    print("Won", end=": ")
                else:
                    print("Dead", end=": ")
                print(steps)
                break

        # progress bar
        percent = (episode + 1) / total_episode
        per_len = int(percent * bar_length)
        a = "*" * per_len
        b = "." * (bar_length - per_len)
        c = percent * 100
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur),
              end="",
              flush=True)

    # end of game
    print("\nGame Over!")
    env.destroy()


if __name__ == "__main__":
    env = Board()
    rl = QLearning(actions=list(range(env.action_num)))

    env.after(100, run)
    env.mainloop()
