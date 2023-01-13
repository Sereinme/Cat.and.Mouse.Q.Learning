# Reinforcement Learning Visualization
# Cat and Mouse game

import time
import tkinter as tk

import numpy as np

ROWS = 4  # grid height
COLUMNS = 4  # grid width
UNIT = int(min(100, 1000 / max(ROWS, COLUMNS)))  # pixels
BOARD_HEIGHT = UNIT * ROWS  # board height
BOARD_WIDTH = UNIT * COLUMNS  # board width
OBSTACLE_NUM = 2  # number of obstacles
PADDING = UNIT * 0.4
GRID_CENTER = np.array([UNIT / 2, UNIT / 2])


class Board(tk.Tk, object):
    """
    Visualized game board environment of
    cat and mouse problem
    """

    def __init__(self):
        """
        Initialize environment variables
        """
        super(Board, self).__init__()

        self.action_space = ["up", "down", "left", "right"]
        self.action_num = len(self.action_space)

        self.title("Cat and Mouse Reinforcement")
        self.geometry("{0}x{1}".format(BOARD_WIDTH, BOARD_HEIGHT))
        self.resizable(False, False)
        self._build_board()

    def _build_board(self):
        """
        Building game board with tkinter
        """
        self.canvas = tk.Canvas(self,
                                bg="white",
                                height=BOARD_HEIGHT,
                                width=BOARD_WIDTH)

        # grid
        for c in range(0, BOARD_WIDTH, UNIT):
            x0, y0, x1, y1 = c, 0, c, BOARD_HEIGHT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, BOARD_HEIGHT, UNIT):
            x0, y0, x1, y1 = 0, r, BOARD_WIDTH, r
            self.canvas.create_line(x0, y0, x1, y1)

        # grid center

        # cat
        cat_index = self._create_cat()

        # mouse
        mouse_index = self._create_mouse()

        # obstacle
        self._create_obstacle(cat_index, mouse_index)

        # pack all
        self.canvas.pack()

    def _create_obstacle(self, cat_index, mouse_index):
        """
        Create obstacles in board with black rectangles

        Parameters:
            cat_index   - initial cat position
            mouse_index - initial mouse position
        """
        self.obstacles = [0 for _ in range(OBSTACLE_NUM)]
        self.obstacles_index = list()
        for i in range(OBSTACLE_NUM):
            while True:
                rd_index = np.array(
                    [np.random.randint(COLUMNS),
                     np.random.randint(ROWS)])
                if (rd_index != cat_index).any() and (
                        rd_index != mouse_index
                ).any() and rd_index.tolist() not in self.obstacles_index:
                    break

            obstacle_pos = GRID_CENTER + rd_index * UNIT
            self.obstacles[i] = self.canvas.create_rectangle(
                obstacle_pos[0] - PADDING,
                obstacle_pos[1] - PADDING,
                obstacle_pos[0] + PADDING,
                obstacle_pos[1] + PADDING,
                fill="black")
            self.obstacles_index.append(rd_index.tolist())

    def _create_cat(self):
        """
        Create cat in board with red rectangles
        """
        cat_index = np.array([0, 0])
        cat_pos = GRID_CENTER + cat_index * UNIT
        self.cat = self.canvas.create_rectangle(cat_pos[0] - PADDING,
                                                cat_pos[1] - PADDING,
                                                cat_pos[0] + PADDING,
                                                cat_pos[1] + PADDING,
                                                fill="crimson",
                                                outline="crimson")
        return cat_index

    def _create_mouse(self):
        """
        Create mouse in board with green circle
        """
        mouse_index = np.array([COLUMNS - 1, ROWS - 1])
        mouse_pos = GRID_CENTER + mouse_index * UNIT
        self.mouse = self.canvas.create_oval(mouse_pos[0] - PADDING,
                                             mouse_pos[1] - PADDING,
                                             mouse_pos[0] + PADDING,
                                             mouse_pos[1] + PADDING,
                                             fill="seagreen",
                                             outline="seagreen")
        return mouse_index

    def _mouse_wander(self):
        """
        Random mouse wandering
        """
        mouse_pos = self.canvas.coords(self.mouse)
        mouse_index = pos2index(mouse_pos)
        cat_index = pos2index(self.canvas.coords(self.cat))
        wander = np.random.randint(self.action_num)
        wander_index = [0, 0]
        wander_move = [0, 0]

        if wander == 0:  # up
            if mouse_pos[1] < UNIT:
                return
            else:
                wander_index = [mouse_index[0], mouse_index[1] - 1]
                wander_move = [0, -1]
        elif wander == 1:  # down
            if mouse_pos[1] > (ROWS - 1) * UNIT:
                return
            else:
                wander_index = [mouse_index[0], mouse_index[1] + 1]
                wander_move = [0, 1]
        elif wander == 2:  # right
            if mouse_pos[0] > (COLUMNS - 1) * UNIT:
                return
            else:
                wander_index = [mouse_index[0] + 1, mouse_index[1]]
                wander_move = [1, 0]
        elif wander == 3:  # left
            if mouse_pos[0] < UNIT:
                return
            else:
                wander_index = [mouse_index[0] - 1, mouse_index[1]]
                wander_move = [-1, 0]
        else:
            print("Unexpected wander!")

        if wander_index in self.obstacles_index or wander_index == cat_index:
            return
        else:
            self.canvas.move(self.mouse, wander_move[0] * UNIT,
                             wander_move[1] * UNIT)

    def reset(self):
        """
        Reset the game board when start a new episode
        """
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.cat)
        self.canvas.delete(self.mouse)
        self._create_mouse()
        self._create_cat()
        return self.canvas.coords(self.cat)

    def step(self, action):
        """
        Cat take one step with action

        Parameters:
            action  - what action cat will use in this state 

        Returns:
            reward  - reward get base on Q(s,a)
            done    - whether this episode is finished or not
            s_      - next cat state/position
            won     - whether catch the mouse
        """
        s = self.canvas.coords(self.cat)
        cat_move = np.array([0, 0])

        if action == 0:  # up
            if s[1] > UNIT:
                cat_move[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (ROWS - 1) * UNIT:
                cat_move[1] += UNIT
        elif action == 2:  # right
            if s[0] < (COLUMNS - 1) * UNIT:
                cat_move[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                cat_move[0] -= UNIT
        else:
            print("Unexpected action!")

        self.canvas.move(self.cat, cat_move[0], cat_move[1])
        s_ = self.canvas.coords(self.cat)

        s_, reward, done, won = self.rewards(s_)
        return s_, reward, done, won

    def rewards(self, s_):
        """
        Determine reward based on next position

        Parameters:
            s_      - cat's next state/position

        Returns:
            reward  - reward get base on Q(s,a)
            done    - whether this episode is finished or not
            s_      - next cat state/position
            won     - whether catch the mouse
        """
        obstacle_region = []
        for i in range(OBSTACLE_NUM):
            obstacle_region.append(self.canvas.coords(self.obstacles[i]))

        if s_ == self.canvas.coords(self.mouse):
            reward = 10
            done = True
            won = True
            s_ = "terminal"
        elif s_ in obstacle_region:
            reward = -10
            done = True
            won = False
            s_ = "terminal"
        else:
            reward = -1
            won = False
            done = False

        return s_, reward, done, won

    def render(self):
        """
        Render game board
        """
        time.sleep(0.01)
        self._mouse_wander()
        self.update()

    def set_title(self, title_name):
        """
        Reset window title name

        Parameters:
            title_name - new title name 
        """
        self.title(title_name)


def pos2index(pos):
    """
    Convert canvas position to board index

    Parameters:
        pos - canvas position from coords

    Returns:
        board index
    """
    x = (pos[0] + pos[2]) / 2
    y = (pos[1] + pos[3]) / 2
    x -= UNIT / 2
    y -= UNIT / 2
    x_index = int(x / UNIT)
    y_index = int(y / UNIT)
    return [x_index, y_index]


def test():
    """
    Test game board
    """
    for _ in range(20):
        _ = env.reset()
        while True:
            env.render()
            a = np.random.randint(4)
            print(a)
            _, _, done, _ = env.step(a)
            if done:
                break


if __name__ == "__main__":
    env = Board()
    env.after(100, test)
    env.mainloop()
