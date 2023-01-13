# Cat.and.Mouse.Q.Learning
Reinforcement learning with cat and mouse game using Q-learning

## Overview

Use `numpy` and `tkinter` to complete Q-learning. 

## Progress Bar

Customized progress bar in reinforcement learning only using `print`, more in `src/main.py`. 

`"\r"` in `print` main start stdout print from head of a line, and it can overwrite the exist words. 

```python
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
```

## Reference
- [Q_Learning_maze](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/2_Q_Learning_maze)
