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
            step_reward=0,
            goal_reward=1,
            cliff_reward=-100,
            walls=None,
            cliffs=None,
            wind=None,
            model_based=False,
            render_mode=None):

        super().__init__()

        self.width = width
        self.height = height

        self._start_pos = (start_pos[0] % width, start_pos[1] % height)
        self._goal_pos = (goal_pos[0] % width, goal_pos[1] % height)

        self._walls = set() if walls is None else set(walls)
        self._cliffs = set() if cliffs is None else set(cliffs)
        self._wind = (0,) * width if wind is None else tuple(wind)

        self._step_reward = step_reward
        self._goal_reward = goal_reward
        self._cliff_reward = cliff_reward

        self._state_positions = []  # state -> pos
        self._state_map = dict()    # pos -> state
        state = 0
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                if pos not in self._walls and pos not in self._cliffs:
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

        self._current_state = None

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
            self._current_state = self.position_to_state(self._start_pos)

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
        pos = self.state_to_position(state)
        if pos == self._goal_pos:
            # Episode was terminated previously
            return state, 0, True

        x, y = pos
        dx, dy = self.actions[action]
        dy -= self._wind[x]
        x = min(max(0, x + dx), self.width - 1)
        y = min(max(0, y + dy), self.height - 1)
        pos = (x, y)

        if pos in self._walls:
            # stay, step reward, not terminated
            return state, self._step_reward, False
        elif pos in self._cliffs:
            # stay, cliff reward, terminated
            return state, self._cliff_reward, True
        elif pos == self._goal_pos:
            # move, goal reward, terminated
            state = self.position_to_state(pos)
            return state, self._goal_reward, True
        else:
            # move, step reward, not terminated
            state = self.position_to_state(pos)
            return state, self._step_reward, False

    def get_action_meaning(self, action):
        return f'{action} ({self.action_meanings[action]})'

    def render_state(self,
                     state,
                     mode,
                     value_fn=None,
                     action_value_fn=None,
                     optimal_action=True,
                     min_value=-1,
                     max_value=1):
        supported_modes = ['ansi', 'rgb_array']
        if mode not in supported_modes:
            supported_modes = ', '.join(f'"{mode}"' for mode in supported_modes)
            raise ValueError(f'Unknown render mode: "{mode}" (must be one of {supported_modes})')

        width = self.width
        height = self.height
        player_pos = self.state_to_position(state)

        if mode == 'ansi':
            result = ''
            for y in range(height):
                if y > 0:
                    result += '\n'

                for x in range(width):
                    if x > 0:
                        result += ' '

                    pos = (x, y)
                    if pos == player_pos:
                        char = 'X'
                    elif pos == self._start_pos:
                        char = 'S'
                    elif pos == self._goal_pos:
                        char = 'G'
                    elif pos in self._walls:
                        char = 'O'
                    elif pos in self._cliffs:
                        char = 'C'
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
                size = 80
                dpi = 300
                fig = plt.figure(figsize=((width * size + 1) / dpi, (height * size + 1) / dpi), dpi=dpi, frameon=False)
                fig.draw(fig.canvas.get_renderer())
                # identity transform + move 1 pixel up
                artist_trans = matplotlib.transforms.Affine2D.from_values(1, 0, 0, 1, 0, 1)

                artists = []

                def fig_draw_artists():
                    nonlocal artists
                    for artist in sorted(artists, key=lambda a: a.zorder):
                        fig.draw_artist(artist)
                    artists = []

                def fig_pixels_to_points(pixels):
                    return 72 / fig.dpi * pixels

                def fig_draw_line(x1, y1, x2, y2, linewidth, color):
                    artist = matplotlib.lines.Line2D((x1, x2), (y1, y2), linewidth=fig_pixels_to_points(linewidth), color=color, transform=artist_trans)
                    artists.append(artist)

                def fig_fill_rect(x, y, width, height, color, zorder=None):
                    artist = matplotlib.patches.Rectangle((x, y), width, height, linewidth=0, facecolor=color, transform=artist_trans)
                    if zorder is not None:
                        artist.zorder = zorder
                    artists.append(artist)

                def fig_fill_circle(x, y, radius, color, zorder=None):
                    artist = matplotlib.patches.Circle((x, y), radius, linewidth=0, facecolor=color, transform=artist_trans)
                    if zorder is not None:
                        artist.zorder = zorder
                    artists.append(artist)

                def fig_draw_text(x, y, text, color, fontsize, ha='center'):
                    artist = matplotlib.text.Text(x, y, text, color, fontsize=fontsize,
                                                  ha=ha, va='center_baseline', transform=artist_trans)
                    artist.figure = fig
                    artists.append(artist)

                redraw_background = not hasattr(self, '_cached_bg')
                redraw_labels = not hasattr(self, '_cached_labels')

                v = None
                if value_fn is None or value_fn(0) is None:
                    if getattr(self, '_cached_v', None) is not None:
                        redraw_labels = True
                    self._cached_v = None
                    v = None
                else:
                    v = np.zeros(self.num_states)
                    for s in range(self.num_states):
                        x = value_fn(s)
                        assert x is not None
                        v[s] = x
                    if getattr(self, '_cached_v', None) is None or not np.all(v == self._cached_v):
                        redraw_labels = True
                    self._cached_v = v
                
                q = None
                if action_value_fn is None or action_value_fn(0, 0) is None:
                    if getattr(self, '_cached_q', None) is not None:
                        redraw_labels = True
                    self._cached_q = None
                    q = None
                else:
                    q = np.zeros((self.num_states, self.num_actions))
                    for s in range(self.num_states):
                        for a in range(self.num_actions):
                            x = action_value_fn(s, a)
                            assert x is not None
                            q[s, a] = x
                    if getattr(self, '_cached_q', None) is None or not np.all(q == self._cached_q):
                        redraw_labels = True
                    self._cached_q = q

                def fill_cell(x, y, xp=0, yp=0, wp=1, hp=1, color=(0, 0, 0), zorder=None):
                    fig_fill_rect((x + xp) * size + 0.5, (height - y + yp - 1) * size + 0.5,
                                  (wp * size) - 1, (hp * size) - 1, color=color, zorder=zorder)

                def draw_cell_text(x, y, text, fontscale, xp, yp, color=(0, 0, 0), ha='center'):
                    fontsize = fig_pixels_to_points(fontscale * size)
                    fig_draw_text((x + xp) * size, (height - y - yp) * size, text, color=color, fontsize=fontsize, ha=ha)

                bg_color = (1, 1, 1)
                grid_color = (0, 0, 0)
                wall_color = (0.35, 0.35, 0.35)
                cliff_color = (0.5, 0.35, 0.35)
                neutral_value_color = (0.9, 0.9, 0.9)
                max_value_color = (67 / 255, 160 / 255, 71 / 255)
                min_value_color = (229 / 255, 57 / 255, 53 / 255)

                if redraw_background:
                    # fill background
                    fig_fill_rect(0, -1, width * size + 1, height * size + 1, bg_color)

                    for y in range(height + 1):
                        fig_draw_line(0, y * size, width * size, y * size, linewidth=1, color=grid_color)
                    for x in range(width + 1):
                        fig_draw_line(x * size, 0, x * size, height * size, linewidth=1, color=grid_color)

                    for pos in self._walls:
                        fill_cell(*pos, color=wall_color)
                    
                    for pos in self._cliffs:
                        fill_cell(*pos, color=cliff_color)

                    fig_draw_artists()
                    self._cached_bg = fig.canvas.copy_from_bbox(fig.bbox)

                if redraw_labels:
                    fig.canvas.restore_region(self._cached_bg)
                    
                    def draw_label(x, y, label):
                        draw_cell_text(x, y, label, fontscale=0.2, xp=0.5, yp=0.85)

                    def draw_reward(x, y, reward, color=(0.3, 0.3, 0.3)):
                        draw_cell_text(x, y, str(reward), fontscale=0.15, xp=0.925, yp=0.85, color=color, ha='right')

                    for pos in self._cliffs:
                        draw_reward(*pos, self._cliff_reward, color=(0, 0, 0))

                    def get_value_color(x):
                        if x < 0:
                            mix = max(min(x, 0.0), min_value) / min_value
                            mix_color = min_value_color
                        else:
                            mix = max(min(x, max_value), 0.0) / max_value
                            mix_color = max_value_color

                        color = tuple(neutral_value_color[i] * (1 - mix) + mix_color[i] * mix for i in range(3))
                        return color
                    
                    bs = 0.12  # box size

                    if v is not None:
                        for (x, y), s in self._state_map.items():
                            state_value = v[s]

                            # fill_cell(x, y, color=get_value_color(state_value))
                            fill_cell(x, y, 0.5 - bs / 2, 0.5 - bs / 2, bs, bs, color=get_value_color(state_value))
                            draw_cell_text(x, y, f'{state_value:.3f}', fontscale=0.15, xp=0.5, yp=0.175)

                    if q is not None:
                        for (x, y), s in self._state_map.items():
                            fill_cell(x, y, 0.5 - bs * 3 / 2, 0.5 - bs / 2, bs, bs, color=get_value_color(q[s, 0]))  # left
                            fill_cell(x, y, 0.5 + bs / 2, 0.5 - bs / 2, bs, bs, color=get_value_color(q[s, 0]))      # right
                            fill_cell(x, y, 0.5 - bs / 2, 0.5 + bs / 2, bs, bs, color=get_value_color(q[s, 0]))      # up
                            fill_cell(x, y, 0.5 - bs / 2, 0.5 - bs * 3 / 2, bs, bs, color=get_value_color(q[s, 0]))  # down

                            if optimal_action:
                                max_q = np.max(q[s])
                                dx = 0.0
                                dy = 0.0
                                num_max = 0
                                for a in range(self.num_actions):
                                    if np.isclose(q[s, a], max_q):
                                        dx_, dy_ = self.actions[a]
                                        dx += dx_
                                        dy += dy_
                                        num_max += 1

                                if num_max > 0:
                                    norm = abs(dx * dx + dy * dy)
                                    if norm > 0:
                                        dx /= norm
                                        dy /= norm
                                    fig_fill_circle((x + 0.5 + dx * bs) * size, (height - y - 0.5 - dy * bs) * size,
                                                    radius=0.03 * size, color=(0, 0, 0), zorder=20)

                    for pos in self._state_map.keys():
                        if pos != self._goal_pos:
                            draw_reward(*pos, self._step_reward)

                    draw_label(*self._start_pos, 'S')
                    draw_label(*self._goal_pos, 'G')
                    draw_reward(*self._goal_pos, self._goal_reward)

                    fig_draw_artists()
                    self._cached_labels = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    fig.canvas.restore_region(self._cached_labels)

                draw_cell_text(*player_pos, 'X', fontscale=0.4, xp=0.5, yp=0.5)
                fig_draw_artists()

                fig.canvas.blit(fig.bbox)
                img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                plt.close(fig)
                return np.asarray(img)

        raise NotImplementedError()


def default_5x5_maze(model_based=False):
    return MazeEnv(width=5, height=5, model_based=model_based,
                   start_pos=(0, 0), goal_pos=(-1, -1),
                   step_reward=0, goal_reward=1,
                   walls=[(2, 0), (2, 2), (2, 3)])


def default_8x8_maze(model_based=False):
    return MazeEnv(width=8, height=8, model_based=model_based,
                   start_pos=(0, 0), goal_pos=(-1, -1),
                   step_reward=0, goal_reward=1,
                   walls=[(2, 0), (2, 2), (2, 3), (4, 7), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4), (6, 3)])


def windy_gridworld(model_based=False, step_reward=-1, goal_reward=0):
    return MazeEnv(width=10, height=7, model_based=model_based,
                   start_pos=(0, 3), goal_pos=(7, 3),
                   step_reward=step_reward, goal_reward=goal_reward,
                   wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])


def cliff_walking(model_based=False):
    return MazeEnv(width=12, height=4, model_based=model_based, start_pos=(0, -1), goal_pos=(-1, -1),
                   step_reward=-1, goal_reward=0, cliff_reward=-100,
                   cliffs=[(1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3)])


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

    def render_state(self, index, mode='ansi', value_fn=None, optimal_action=None, min_value=-1, max_value=1):
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


class BlackjackEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.env = gym.make('Blackjack-v1', sab=True)

        num_states = 1
        dims = []
        for space in self.env.observation_space.spaces:
            num_states *= space.n
            dims.append(space.n)
        self.dims = tuple(dims)
        
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)

    def _state_to_index(self, state):
        return int(np.ravel_multi_index(state, self.dims))

    def _index_to_state(self, index):
        state = np.unravel_index(index, self.dims)
        return tuple(int(x) for x in state)

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        return self._state_to_index(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return self._state_to_index(state), reward, terminated, truncated, info
    
    def render_state(self, state, **kwargs):
        state = self._index_to_state(state)
        return f'Player\'s sum: {state[0]}\nDealer\'s card: {state[1]}\nUsable ace: {"yes" if state[2] == 1 else "no"}'
    
    def get_action_meaning(self, action):
        return ['0 (stick)', '1 (hit)'][action]
