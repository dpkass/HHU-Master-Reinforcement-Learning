# author: Jan Robine
# date:   2023-04-25
# course: reinforcement learning

# this gui is designed for the jupyter notebook

import contextlib
import functools
import ipywidgets as widgets
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum
from IPython.display import display, clear_output
from typing import Any, Optional, Sequence


def create_renderer(env, fps=60, figsize=None):
    import matplotlib.pyplot as plt
    env.reset()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')
    img = ax.imshow(env.render())
    fig.suptitle('')
    fig.tight_layout()

    def render(title=''):
        frame = env.render()
        fig.suptitle(title)
        img.set_data(frame)
        display(fig)
        clear_output(wait=True)
        time.sleep(1 / fps)

    return render


class RLGui(widgets.VBox):

    def __init__(self,
                 env,
                 agents=None,
                 max_steps=None,
                 episode=None,
                 enable_interaction=True,
                 enable_user_actions=True,
                 enable_reset=True,
                 enable_optimization=True,
                 render_mode=None,
                 render_optimal_action=True,
                 render_min_value=-1,
                 render_max_value=1,
                 interval=150,
                 print_state=False):

        super().__init__(layout=dict(padding='0px 8px 4px 8px'))

        self.env = env
        self.agents = dict() if agents is None else agents
        self.max_steps = max_steps
        self.enable_interaction = enable_interaction
        self.enable_user_actions = enable_user_actions
        self.enable_reset = enable_reset
        self.enable_optimization = enable_optimization
        self.render_mode = render_mode
        self.render_optimal_action = render_optimal_action
        self.render_min_value = render_min_value
        self.render_max_value = render_max_value
        self.interval = interval
        self.print_state = print_state

        if episode is not None:
            self.states =  list(episode['states'])
            self.actions = list(episode['actions'])
            self.rewards = list(episode['rewards'])
            self.terminated = episode['terminated']
            self.truncated  = episode['truncated']
            assert len(self.states) == len(self.actions) + 1
            assert len(self.actions) == len(self.rewards)
        else:
            state, info = env.reset()
            self.states = [state]
            self.actions = []
            self.rewards = []
            self.terminated = False
            self.truncated = False

        self.optim_agent = None
        self.optim_generator = None
        self.optim_init = None
        self.optim_params = None
        self.optim_cmd = None

        self.step_guard = False
        self.was_done = False

        # Widgets
        self.episode_output = widgets.Output()

        self.reset_button = widgets.Button(description='Reset')
        self.reset_button.on_click(lambda _: self.on_click_reset())

        self.truncate_button = widgets.Button(description='Truncate')
        self.truncate_button.on_click(lambda _: self.on_click_truncate())

        self.episode_hbox = widgets.HBox([self.reset_button, self.truncate_button])
        self.episode_hbox.add_class('margin_bottom_16')

        self.render_output = widgets.Output()
        self.step_output = widgets.Output()

        # agent
        agent_keys = tuple(self.agents.keys())
        self.agent_buttons = widgets.ToggleButtons(options=agent_keys, layout=dict(display='inline'))
        self.agent_buttons.observe(lambda change: self.render(), names=['index'])
        self.agent_hbox = self._hbox_with_label('Agent:', self.agent_buttons)

        # interact
        self.interact_play = _InfinitePlay(interval=interval, layout=dict(display='inline'))
        self.interact_play.observe(lambda change: self.on_step_play(), names=['value'])
        self.interact_play.observe(lambda change: self.render(), names=['playing'])
        self.interact_hbox = self._hbox_with_label('Interact:', self.interact_play)
        if len(self.agents) > 0:
            self.interact_hbox.add_class('margin_top_8')

        user_action_buttons = []
        if self.enable_user_actions:
            for action in range(env.action_space.n):
                button = widgets.Button(description=self._get_action_meaning(action), layout=dict(display='inline'))
                button.on_click(lambda _, action=action: self.on_click_action(action))
                user_action_buttons.append(button)
        self.user_action_hbox = widgets.HBox(user_action_buttons)

        self.interact_vbox = widgets.VBox([
            self.interact_hbox,
            self.user_action_hbox
        ])

        # optim
        self.optim_params_vbox = widgets.VBox([])
        self.optim_params_vbox.add_class('margin_top_8')

        self.optim_play = _InfinitePlay(interval=interval, layout=dict(display='inline'))
        self.optim_play.observe(lambda change: self.on_step_optim(), names=['value'])
        self.optim_play.observe(lambda change: self.render(), names=['playing'])

        self.optim_info_label = self._custom_label('')
        self.optim_info_label.add_class('margin_left_4')

        self.optim_hbox = self._hbox_with_label('Optimize:', self.optim_play, self.optim_info_label)
        self.optim_hbox.add_class('margin_top_8')

        self.optim_option_hbox = widgets.HBox([])

        self.optim_vbox = widgets.VBox([
            self.optim_params_vbox,
            self.optim_hbox,
            self.optim_option_hbox
        ])

        # replay
        self.replay_play = _StoppablePlay(value=0, min=0, max=1, step=1, interval=interval, layout=dict(display='inline'))
        self.replay_play.observe(lambda change: self.on_observe_replay(), names=['value'])
        self.interact_play.observe(lambda change: self.render(), names=['playing'])
        self.replay_hbox = self._hbox_with_label('Replay:', self.replay_play)

        self.step_slider = widgets.IntSlider(value=0, min=0, max=1, step=1)
        self.step_hbox = self._hbox_with_label('Step:', self.step_slider)

        self.replay_vbox = widgets.VBox([
            self.replay_hbox,
            self.step_hbox
        ])

        widgets.link((self.step_slider, 'value'), (self.replay_play, 'value'))

        styles = '<style>' \
            '.margin_left_4 { margin-left: 4px } ' \
            '.margin_right_4 { margin-right: 4px } ' \
            '.margin_top_8 { margin-top: 8px } ' \
            '.margin_bottom_16 { margin-bottom: 16px } ' \
            '.play_infinite { width: 70px } ' \
            '.play_infinite button:nth-child(2) { display: none } ' \
            '.play_infinite button:nth-child(1) { width: 70px } ' \
            '.improved_input input { max-width: 80px !important } ' \
            '.improved_input label { width: 100px !important; text-align: left !important } ' \
            '</style>'

        self.children = (
            widgets.HTML(styles),
            self.episode_output,
            self.episode_hbox,
            self.render_output,
            self.step_output,
            self.agent_hbox,
            self.interact_vbox,
            self.optim_vbox,
            self.replay_vbox
        )

        self.render()

    @staticmethod
    def _custom_label(text, classes=tuple(), layout=None):
        if layout is None:
            layout = dict()
        label = widgets.Label(value=text, layout=dict(display='inline', **layout))
        for class_ in classes:
            label.add_class(class_)
        return label

    @staticmethod
    def _hbox_with_label(text, *children):
        label = RLGui._custom_label(text, classes=['margin_right_4'])
        return widgets.HBox([label, *children])

    @contextlib.contextmanager
    def _guard_step(self):
        self.step_guard = True
        yield
        self.step_guard = False

    def _get_action_meaning(self, action):
        if hasattr(self.env, 'get_action_meaning'):
            return self.env.get_action_meaning(action)
        return str(action)

    def render(self):
        episode_length = len(self.rewards)

        with self.episode_output:
            clear_output(wait=True)
            print(f'Episode length: {len(self.rewards):>10}')
            print(f'Sum of rewards: {sum(self.rewards):>10}')
            if self.terminated:
                print('Episode status: terminated')
            elif self.truncated:
                print('Episode status:  truncated')
            else:
                print('Episode status:    running')

        if not (self.terminated or self.truncated):
            step = episode_length
        else:
            step = self.replay_play.value

        state = self.states[step]
        action = self.actions[step - 1] if step > 0 else None
        reward = self.rewards[step - 1] if step > 0 else None

        selected_agent = self.agent_buttons.value
        agent = self.agents.get(selected_agent, None)

        with self.render_output:
            clear_output(wait=True)
            if self.render_mode is not None:
                from PIL import Image
                value_fn = None if agent is None else agent.value
                action_value_fn = None if (agent is None or not hasattr(agent, 'action_value')) else agent.action_value
                render_result = self.env.render_state(state, mode=self.render_mode, value_fn=value_fn, action_value_fn=action_value_fn,
                                                      optimal_action=self.render_optimal_action, min_value=self.render_min_value, max_value=self.render_max_value)
                if isinstance(render_result, str):
                    print(render_result)
                    print()
                elif isinstance(render_result, np.ndarray):
                    img = Image.fromarray(render_result)
                    display(img)
                elif isinstance(render_result, Image.Image):
                    display(render_result)
                else:
                    raise TypeError(f'Unsupported render type: {type(render_result)}')

        with self.step_output:
            clear_output(wait=True)
            if self.print_state:
                print(f'State: {state}')
            print(f'Action: {"-" if action is None else self._get_action_meaning(action)}')
            print(f'Reward: {"-" if reward is None else reward}')

        show_widgets = []
        hide_widgets = []

        waiting_for_episode = self.optim_cmd is not None and isinstance(self.optim_cmd, RLCmd.WaitForEpisode)
        if (self.terminated or self.truncated) and not waiting_for_episode:
            hide_widgets.extend([self.agent_hbox, self.interact_vbox, self.optim_vbox])
            self.optim_play.stop()

            if not self.was_done:
                # episode just finished
                self.step_slider.max = episode_length
                self.replay_play.max = episode_length
                self.step_slider.value = 0
                self.replay_play.stop()
                self.interact_play.stop()

            hide_widgets.append(self.truncate_button)
            if self.enable_reset:
                show_widgets.append(self.episode_hbox)
            else:
                hide_widgets.append(self.episode_hbox)
            
            show_widgets.append(self.replay_vbox)

            self.agent_buttons.disabled = False
            self.reset_button.disabled = self.replay_play.playing
        else:
            show_widgets.extend([self.episode_hbox, self.truncate_button])
            hide_widgets.append(self.replay_vbox)

            if len(self.agents) <= 1:
                hide_widgets.append(self.agent_hbox)
            else:
                show_widgets.append(self.agent_hbox)

            if self.enable_interaction:
                show_widgets.append(self.interact_vbox)

                if len(self.agents) == 0:
                    hide_widgets.append(self.interact_play)
                else:
                    show_widgets.extend([self.interact_hbox, self.interact_play])
                
                if self.enable_user_actions:
                    show_widgets.append(self.user_action_hbox)
                else:
                    hide_widgets.append(self.user_action_hbox)
            else:
                hide_widgets.append(self.interact_vbox)

            if len(self.agents) == 0:
                hide_widgets.extend([self.optim_vbox])
                self.optim_play.stop()
            else:
                self.setup_optimization()

                if self.optim_generator is not None and self.optim_cmd is not None:
                    show_widgets.append(self.optim_vbox)

                    if len(self.optim_init.params) == 0:
                        hide_widgets.append(self.optim_params_vbox)  # just for spacing
                    else:
                        show_widgets.append(self.optim_params_vbox)

                    cmd = self.optim_cmd
                    if isinstance(cmd, RLCmd.Wait):
                        hide_widgets.append(self.optim_play)
                        self.optim_play.stop()
                        self.optim_info_label.value = str(cmd.message) if cmd.message is not None else ''
                        for button in self.optim_option_hbox.children:
                            button.disabled = True

                    elif isinstance(cmd, RLCmd.WaitForOption):
                        if cmd.step is not None:
                            show_widgets.append(self.optim_play)
                        else:
                            hide_widgets.append(self.optim_play)
                            self.optim_play.stop()

                        self.optim_info_label.value = str(cmd.message) if cmd.message is not None else ''

                        is_playing = self.interact_play.playing or self.optim_play.playing
                        for option, button in zip(self.optim_init.options.keys(), self.optim_option_hbox.children):
                            if is_playing:
                                button.disabled = True
                            else:
                                button.disabled = option not in cmd.active

                    elif isinstance(cmd, RLCmd.WaitForEpisode):
                        hide_widgets.append(self.optim_play)
                        self.optim_play.stop()
                        self.optim_info_label.value = 'Collecting episode...'
                        for button in self.optim_option_hbox.children:
                            button.disabled = True
                    else:
                        raise TypeError(type(cmd))
                else:
                    hide_widgets.extend([self.optim_vbox])
                    self.optim_play.stop()

            self.optim_play.disabled = self.interact_play.playing
            self.interact_play.disabled = self.optim_play.playing

            is_playing = self.interact_play.playing or self.optim_play.playing
            self.agent_buttons.disabled = is_playing
            self.reset_button.disabled = is_playing
            self.truncate_button.disabled = is_playing
            for button in self.user_action_hbox.children:
                button.disabled = is_playing

        for widget in hide_widgets:
            widget.layout.display = 'none'

        for widget in show_widgets:
            if isinstance(widget, widgets.HBox):
                widget.layout.display = 'flex'
            else:
                widget.layout.display = 'block'
        
        self.was_done = self.terminated or self.truncated

    def reset(self):
        state, info = self.env.reset()
        self.states = [state]
        self.actions = []
        self.rewards = []
        self.terminated = False
        self.truncated = False

    def step(self, action, max_steps=None):
        assert not (self.terminated or self.truncated)

        if action is None:
            selected_agent = self.agent_buttons.value
            agent = self.agents.get(selected_agent, None)
            if agent is not None:
                action = agent.policy(self.states[-1])
            else:
                raise ValueError()

        self.actions.append(action)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        self.states.append(next_state)
        self.terminated = terminated
        self.truncated = truncated
        if max_steps is None:
            max_steps = self.max_steps
        if max_steps is not None and len(self.rewards) >= max_steps:
            self.truncated = True

    def truncate(self):
        self.truncated = True

    def get_episode(self):
        return {'states': tuple(self.states), 'actions': tuple(self.actions), 'rewards': tuple(self.rewards),
                'terminated': self.terminated, 'truncated': self.truncated}

    def setup_optimization(self):
        selected_agent = self.agent_buttons.value
        agent = self.agents[selected_agent]

        if self.optim_generator is not None and selected_agent == self.optim_agent:
            return

        self.optim_play.stop()

        if agent is None or not hasattr(agent, 'interactive_optimization'):
            self.optim_generator = None
            self.optim_agent = None
            self.optim_init = None
            self.optim_params = None
            self.optim_cmd = None
            return

        self.optim_agent = selected_agent
        self.optim_generator = agent.interactive_optimization()
        cmd = self.optim_generator.send(None)
        assert isinstance(cmd, RLCmd.Init)
        self.optim_init = RLCmd.Init(options=dict(cmd.options) if cmd.options is not None else dict(),
                                     params=dict(cmd.params) if cmd.params is not None else dict(),
                                     params_callback=cmd.params_callback)

        def replace_children(parent, values, setup_fn):
            # try to reuse widgets if possible
            new_children = []
            changed_children = len(values) != len(parent.children)
            for value, child in zip(values, parent.children):
                new_child = setup_fn(value, child)
                if new_child is not child:
                    changed_children = True
                new_children.append(new_child)
            for i in range(len(parent.children), len(values)):
                new_child = setup_fn(values[i], None)
                new_children.append(new_child)
            if changed_children:
                parent.children = tuple(new_children)

        # params
        params = dict()
        for param, param_info in self.optim_init.params.items():
            init_value = param_info[2]
            params[param] = init_value
        self.optim_params = params

        def setup_param_widget(value, widget=None):
            param, param_info = value
            label, param_type, init_value = param_info[:3]
            queue = list(param_info[3:])

            if param_type == 'float':
                min_value = queue.pop(0)
                max_value = queue.pop(0)
                step = queue.pop(0) if len(queue) > 0 else 1e-4
                description = label + ':'

                if widget is None or not isinstance(widget, widgets.FloatSlider):
                    widget = widgets.FloatSlider(value=init_value, min=min_value, max=max_value,
                                                step=step, description=description, readout_format='.4f')
                    widget.add_class('improved_input')
                else:
                    widget.unobserve_all()

                widget.min = min(widget.min, min_value)
                widget.max = max(widget.max, max_value)
                widget.min = min_value
                widget.max = max_value
                widget.step = step
                widget.value = init_value
                widget.description = description

                widget.observe(functools.partial(self.on_change_optim_param, param=param), names=['value'])

            elif param_type == 'float_log':
                base = queue.pop(0)
                min_exponent = queue.pop(0)
                max_exponent = queue.pop(0)
                step = queue.pop() if len(queue) > 0 else 0.1
                description = label + ':'

                if widget is None or not isinstance(widget, widgets.FloatLogSlider):
                    widget = widgets.FloatLogSlider(readout_format='.4f')
                    widget.add_class('improved_input')
                else:
                    widget.unobserve_all()

                widget.min = min(widget.min, min_exponent)
                widget.max = max(widget.max, max_exponent)
                widget.min = min_exponent
                widget.max = max_exponent
                widget.base = base
                widget.value = init_value
                widget.step = step
                widget.description = description

                widget.observe(functools.partial(self.on_change_optim_param, param=param), names=['value'])

            else:
                raise ValueError(param_type)

            return widget

        replace_children(self.optim_params_vbox, tuple(self.optim_init.params.items()), setup_param_widget)

        # options
        def setup_option_button(value, button=None):
            option, label = value
            if button is None:
                button = widgets.Button(description=label, layout=dict(display='inline'))
            else:
                button._click_handlers.callbacks = []
                button.description = label
            button.on_click(lambda _, option=option: self.on_click_optim_option(option))
            return button

        replace_children(self.optim_option_hbox, tuple(self.optim_init.options.items()), setup_option_button)

        self.optimize(None)

    def optimize(self, result):
        assert self.optim_generator is not None
        try:
            cmd = self.optim_generator.send(result)
            self.optim_cmd = cmd
            if isinstance(cmd, RLCmd.Wait):
                pass
            elif isinstance(cmd, RLCmd.WaitForOption):
                if cmd.step is not None:
                    assert cmd.step in cmd.active
                    self.optim_play.interval = cmd.interval if cmd.interval is not None else self.interval
            elif isinstance(cmd, RLCmd.WaitForEpisode):
                if len(self.states) > 1 or self.terminated or self.truncated:
                    self.reset()
                if cmd.interval == -1:
                    while not (self.terminated or self.truncated):
                        self.step(action=None, max_steps=cmd.max_steps)
                    episode = self.get_episode()
                    self.reset()
                    self.optimize(episode)
                else:
                    self.optim_play.interval = cmd.interval if cmd.interval is not None else self.interval
                    self.optim_play.start()
            elif isinstance(cmd, RLCmd.WaitForStep):
                s = self.states[-1]
                self.step(action=cmd.action, max_steps=cmd.max_steps)
                a = self.actions[-1]
                r = self.rewards[-1]
                next_s = self.states[-1]
                terminated = self.terminated
                truncated = self.truncated
                if terminated or truncated:
                    self.reset()
                self.optimize((s, a, r, next_s, terminated, truncated))
            else:
                raise TypeError(type(cmd))
        except StopIteration as e:
            self.optim_agent = None
            self.optim_generator = None
            self.optim_init = None
            self.optim_params = None
            self.optim_cmd = None

    def on_click_reset(self):
        self.reset()
        self.render()

    def on_click_truncate(self):
        self.truncate()
        self.render()

    def on_click_action(self, action):
        self.step(action)
        self.render()

    def on_change_optim_param(self, change, param):
        value = change['new']
        if value != self.optim_params[param]:
            self.optim_params[param] = value
            callback = self.optim_init.params_callback
            if callback is not None:
                result = callback(self.optim_params)
                if result == RLParamsResult.RESET_GENERATOR:
                    self.optim_generator = None
                    self.setup_optimization()
                    self.render()
                elif result == RLParamsResult.RENDER:
                    self.render()
                elif result is not None:
                    raise ValueError(result)

    def on_click_optim_option(self, option):
        if isinstance(self.optim_cmd, RLCmd.WaitForOption):
            self.optimize(option)
            self.render()

    def on_step_play(self):
        if self.step_guard:
            return

        with self._guard_step():
            if self.terminated or self.truncated:
                self.render()
                return

            self.step(action=None)
            if self.terminated or self.truncated:
                self.step_slider.value = 0
            self.render()

    def on_step_optim(self):
        if self.step_guard:
            return

        with self._guard_step():
            self.setup_optimization()  # ensure correct agent
            if self.optim_generator is not None:
                cmd = self.optim_cmd
                if cmd is not None:
                    if isinstance(cmd, RLCmd.WaitForOption):
                        if cmd.step is not None:
                            self.optimize(cmd.step)
                    elif isinstance(cmd, RLCmd.WaitForEpisode):
                        if not (self.terminated or self.truncated):
                            self.step(action=None, max_steps=cmd.max_steps)
                        if self.terminated or self.truncated:
                            self.optim_play.stop()
                            self.optim_play.interval = self.interval
                            episode = self.get_episode()
                            self.reset()
                            self.optimize(episode)
                    else:
                        raise TypeError(type(cmd))
                self.render()

    def on_observe_replay(self):
        if self.step_guard:
            return
        
        with self._guard_step():
            self.render()


class RLCmd:

    @dataclass
    class Init:
        options: Optional[dict] = None
        params: Optional[dict] = None
        params_callback: Optional[Any] = None

    @dataclass
    class Wait:
        message: Optional[str] = None

    @dataclass
    class WaitForOption:
        active: Sequence
        step: Optional[str] = None
        interval: Optional[int] = None
        message: Optional[str] = None

    @dataclass
    class WaitForEpisode:
        max_steps: Optional[int] = None
        interval: Optional[int] = None

    @dataclass
    class WaitForStep:
        action: Optional[Any]
        max_steps: Optional[int] = None


class RLParamsResult(str, Enum):
    RENDER = 'render'
    RESET_GENERATOR = 'reset_generator'



class _StoppablePlay(widgets.Play):

    def stop(self):
        #self.traits()['playing'].set(self, False)
        if self.playing:
            self.playing = False
            self.value = self.min
    
    def start(self):
        self.playing = True


class _InfinitePlay(_StoppablePlay):
    
    def __init__(self, interval, layout=None):
        super().__init__(value=0, min=0, max=np.iinfo(np.int32).max, step=1, interval=interval, show_repeat=False, layout=layout)
        self.add_class('play_infinite')
