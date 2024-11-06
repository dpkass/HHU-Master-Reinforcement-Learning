# author: Jan Robine
# date:   2023-04-25
# course: reinforcement learning

import gymnasium as gym
import numpy as np


class MazeEnv(gym.Env):

    def __init__(
            self,
            width=5,
            height=5,
            start_pos=(0, 0),
            goal_pos=(-1, -1),
            walls=None,
            model_based=False,
            render_mode=None):

        super().__init__()

        self.width = width
        self.height = height

        self._walls = set() if walls is None else set(walls)
        self._state_positions = []  # state -> pos
        self._state_map = dict()    # pos -> state
        state = 0
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                if pos not in self._walls:
                    self._state_positions.append(pos)
                    self._state_map[pos] = state
                    state += 1
        del state

        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.action_meanings = ('left', 'right', 'up', 'down')

        self.num_states = len(self._state_positions)
        self.num_actions = len(self.actions)
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self._start_state = self.position_to_state(start_pos)
        self._goal_state = self.position_to_state(goal_pos)   # currently only one goal state possible
        self._current_state = self._start_state

        self._model_based = model_based
        if model_based:
            # transition probabilities
            P = np.zeros((self.num_states, self.num_actions, self.num_states))
            # reward function
            R = np.zeros((self.num_states, self.num_actions))

            for state in range(self.num_states):
                for action in range(self.num_actions):
                    next_state, reward, terminated = self.transition(state, action)
                    P[state, action, next_state] = 1
                    R[state, action] = reward

            self._P = P
            self._R = R

        self._render_mode = render_mode

    @property
    def P(self):
        if not self._model_based:
            raise ValueError('Set model_based=True to get the transition probabilities')
        return self._P

    @property
    def R(self):
        if not self._model_based:
            raise ValueError('Set model_based=True to get the reward function')
        return self._R
    
    @property
    def current_state(self):
        return self._current_state

    def reset(self, seed=None, options=None):
        # Reset the state to the start state
        if options is not None and 'state' in options:
            # The user can choose a specific start state
            self._current_state = options['state']
        else:
            self._current_state = self._start_state

        info = dict()
        return self._current_state, info

    def step(self, action):
        state = self._current_state
        next_state, reward, terminated = self.transition(state, action)
        truncated = False
        info = dict()
        self._current_state = next_state
        return next_state, reward, terminated, truncated, info

    def render(self):
        return self.render_state(self._current_state, self._render_mode)

    # Utility functions

    def state_to_position(self, state):
        return self._state_positions[state]

    def position_to_state(self, pos):
        x, y = pos
        x = x % self.width
        y = y % self.height
        pos = (x, y)
        return self._state_map[pos]

    def transition(self, state, action):
        if state == self._goal_state:
            return state, 0, True

        x, y = self.state_to_position(state)
        dx, dy = self.actions[action]
        x = min(max(0, x + dx), self.width - 1)
        y = min(max(0, y + dy), self.height - 1)
        if (x, y) in self._walls:
            return state, 0, False   # STAY, zero reward, not terminated
        state = self.position_to_state((x, y))
        terminated = (state == self._goal_state)
        reward = 1 if terminated else 0
        return state, reward, terminated

    def get_action_meaning(self, action):
        return f'{action} ({self.action_meanings[action]})'

    def render_state(self, state, mode, value_fn=None, optimal_action=False):
        supported_modes = ['ansi', 'rgb_array']
        if mode not in supported_modes:
            supported_modes = ', '.join(f'"{mode}"' for mode in supported_modes)
            raise ValueError(f'Unknown render mode: "{mode}" (must be one of {supported_modes})')

        start_state = self._start_state
        goal_state = self._goal_state
        width = self.width
        height = self.height

        if mode == 'ansi':
            result = ''
            for y in range(height):
                if y > 0:
                    result += '\n'

                for x in range(width):
                    if x > 0:
                        result += ' '

                    pos = (x, y)
                    if pos in self._walls:
                        char = 'O'
                    else:
                        s = self.position_to_state(pos)
                        if s == state:
                            char = 'X'
                        elif s == start_state:
                            char = 'S'
                        elif s == goal_state:
                            char = 'G'
                        else:
                            char = '.'
                    result += char
            return result

        if mode == 'rgb_array':
            import matplotlib
            import matplotlib.lines
            import matplotlib.text
            import matplotlib.transforms
            import matplotlib.patches
            import matplotlib.pyplot as plt
            from PIL import Image

            with plt.ioff():
                size = 64
                dpi = 300
                fig = plt.figure(figsize=((width * size + 1) / dpi, (height * size + 1) / dpi), dpi=dpi, frameon=False)
                fig.draw(fig.canvas.get_renderer())
                # identity transform + move 1 pixel up
                artist_trans = matplotlib.transforms.Affine2D.from_values(1, 0, 0, 1, 0, 1)

                def fig_pixels_to_points(pixels):
                    return 72 / fig.dpi * pixels

                def fig_draw_line(x1, y1, x2, y2, linewidth, color):
                    artist = matplotlib.lines.Line2D((x1, x2), (y1, y2), linewidth=fig_pixels_to_points(linewidth), color=color, transform=artist_trans)
                    fig.draw_artist(artist)

                def fig_fill_rect(x, y, width, height, color):
                    artist = matplotlib.patches.Rectangle((x, y), width, height, linewidth=0, facecolor=color, transform=artist_trans)
                    fig.draw_artist(artist)

                def fig_fill_circle(x, y, radius, color):
                    artist = matplotlib.patches.Circle((x, y), radius, linewidth=0, facecolor=color, transform=artist_trans)
                    fig.draw_artist(artist)

                def fig_draw_text(x, y, text, color, fontsize):
                    artist = matplotlib.text.Text(x, y, text, color=color, fontsize=fontsize,
                                                  ha='center', va='center_baseline', transform=artist_trans)
                    artist.figure = fig
                    fig.draw_artist(artist)

                redraw_background = not hasattr(self, '_cached_bg')
                redraw_labels = not hasattr(self, '_cached_labels')

                values = None
                if value_fn is None or value_fn(0) is None:
                    if getattr(self, '_cached_values', None) is not None:
                        redraw_labels = True
                    self._cached_values = None
                    values = None
                else:
                    values = []
                    for s in range(self.num_states):
                        v = value_fn(s)
                        assert v is not None
                        values.append(v)

                    if getattr(self, '_cached_values', None) is None or \
                        any(v != cv for v, cv in zip(values, self._cached_values)):
                        redraw_labels = True

                    self._cached_values = values

                def fill_cell(x, y, color):
                    fig_fill_rect(x * size + 0.5, (height - y - 1) * size + 0.5, size - 1, size - 1, color=color)

                def draw_cell_text(x, y, text, fontscale, xp, yp, color=(0, 0, 0)):
                    fontsize = fig_pixels_to_points(fontscale * size)
                    fig_draw_text((x + xp) * size, (height - y - yp) * size, text, color=color, fontsize=fontsize)

                bg_color = (1, 1, 1)
                grid_color = (0, 0, 0)
                wall_color = (0.35, 0.35, 0.35)
                max_value_color = (67 / 255, 160 / 255, 71 / 255)

                if redraw_background:
                    # fill background
                    fig_fill_rect(0, -1, width * size + 1, height * size + 1, bg_color)

                    for y in range(height + 1):
                        fig_draw_line(0, y * size, width * size, y * size, linewidth=1, color=grid_color)
                    for x in range(width + 1):
                        fig_draw_line(x * size, 0, x * size, height * size, linewidth=1, color=grid_color)

                    for (x, y) in self._walls:
                        fill_cell(x, y, color=wall_color)
                    
                    self._cached_bg = fig.canvas.copy_from_bbox(fig.bbox)

                if redraw_labels:
                    fig.canvas.restore_region(self._cached_bg)

                    if values is not None:
                        for (x, y), s in self._state_map.items():
                            v = values[s]
                            mix = max(min(v, 1.0), 0.0)
                            color = tuple(bg_color[i] * (1 - mix) + max_value_color[i] * mix for i in range(3))
                            fill_cell(x, y, color)
                            draw_cell_text(x, y, f'{v:.3f}', fontscale=0.2, xp=0.5, yp=0.2)

                            if optimal_action:
                                max_a = None
                                max_v = 0.0
                                for a in range(self.action_space.n):
                                    next_s, _, _ = self.transition(s, a)
                                    next_v = values[next_s]
                                    if next_v > max_v:
                                        max_v = next_v
                                        max_a = a

                                if max_a is not None:
                                    dx, dy = self.actions[max_a]
                                    scale = 0.125
                                    fig_draw_line((x + 0.2) * size, (height - y - 0.8) * size,
                                                  (x + 0.2 + dx * scale) * size, (height - y - 0.8 - dy * scale) * size,
                                                  linewidth=1, color=(0, 0, 0))

                                fig_fill_circle((x + 0.2) * size + 0.5, (height - y - 0.8) * size - 0.5,
                                                radius=2, color=(0, 0, 0))

                    x, y = self.state_to_position(self._start_state)
                    draw_cell_text(x, y, 'S', fontscale=0.3, xp=0.5, yp=0.8)

                    x, y = self.state_to_position(self._goal_state)
                    draw_cell_text(x, y, 'G', fontscale=0.3, xp=0.5, yp=0.8)
                    draw_cell_text(x, y, '+1', fontscale=0.2, xp=0.8, yp=0.8)

                    self._cached_labels = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    fig.canvas.restore_region(self._cached_labels)

                x, y = self.state_to_position(state)
                draw_cell_text(x, y, 'X', fontscale=0.4, xp=0.5, yp=0.5)

                fig.canvas.blit(fig.bbox)
                img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                plt.close(fig)
                return np.asarray(img)

        raise NotImplementedError()


def default_5x5_maze(model_based=False):
    return MazeEnv(width=5, height=5, model_based=model_based, walls=[(2, 0), (2, 2), (2, 3)])


def default_8x8_maze(model_based=False):
    return MazeEnv(width=8, height=8, model_based=model_based,
                   walls=[(2, 0), (2, 2), (2, 3), (4, 7), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4), (6, 3)])


class TicTacToeEnv(gym.Env):
    _shape = (3, 3, 3, 3, 3, 3, 3, 3, 3)

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(19683)  # 3^9
        self.action_space = gym.spaces.Discrete(9)

    @staticmethod
    def state_to_index(state):
        return np.ravel_multi_index(state, TicTacToeEnv._shape)

    @staticmethod
    def index_to_state(index):
        return np.array(np.unravel_index(index, TicTacToeEnv._shape), dtype=np.uint8)

    @staticmethod
    def get_valid_actions(index):
        state = TicTacToeEnv.index_to_state(index)
        return np.arange(9, dtype=int)[state == 0]

    def reset(self):
        self.state = np.zeros(9, dtype=np.uint8)
        info = {}
        return self.state_to_index(self.state), info

    def step(self, action):
        state = self.state

        # Perform action
        if state[action] == 0:
            unique, counts = np.unique(state, return_counts=True)
            count1 = counts[unique == 1] if 1 in unique else 0
            count2 = counts[unique == 2] if 2 in unique else 0
            if count1 > count2:
                state[action] = 2
            else:
                state[action] = 1

        # Determine winning player        
        reward = 0.0
        terminated = False
        for player in (1, 2):
            mask = (state == player).reshape(3, 3)
            if mask.all(axis=0).any() or mask.all(axis=1).any() or \
               mask.diagonal().all() or np.fliplr(mask).diagonal().all():
                reward = 1.0 if player == 1 else 0.0
                terminated = True
                break
        
        # Check for draw
        if not terminated and np.all(state != 0):
            reward = 0.0
            terminated = True

        truncated = False
        info = {}
        return self.state_to_index(state), reward, terminated, truncated, info
    
    def render_state(self, index, mode='ansi', value_fn=None, optimal_action=None):
        assert mode == 'ansi'
        state = self.index_to_state(index)
        result = []
        for i in range(3):
            row = ''
            for j in range(3):
                player = state[i * 3 + j]
                if player == 1:
                    row += 'X'
                elif player == 2:
                    row += 'O'
                else:
                    row += '-'
            result.append(row)
        return '\n'.join(result)
