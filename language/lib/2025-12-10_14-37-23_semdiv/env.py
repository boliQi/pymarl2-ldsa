import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.reward = 0
        self.additional_reward = 0
        self.current_step = None

    def setup(self, position, level, field_size):
        self.position = position
        self.level = level
        self.field_size = field_size

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    # action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "reward", "additional_reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        additional_reward_id=0,
    ):

        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.return_all_llm_rewards = force_coop
        self._game_over = None

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.viewer = None

        self.n_agents = len(self.players)

        self.additional_reward_lst = [
            self.no_additional_reward,
            self.prefer_food_A,
            self.prefer_food_B,
            self.prefer_food_C,
            self.prefer_food_D,
        ]

        for i in range(30):
            try:
                self.additional_reward_lst.append(getattr(self, f'llm{i+1}'))
            except AttributeError:
                pass

        self.compute_additional_reward = self.additional_reward_lst[additional_reward_id]

        self.agents_position = {"1": np.array([0, 0]), "2": np.array([0, 0])}
        self.foods_position = {"A": np.array([0, 0]),
                               "B": np.array([0, field_size[0] - 1]),
                               "C": np.array([field_size[0] - 1, 0]),
                               "D": np.array([field_size[0] - 1, field_size[0] - 1])}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_food = self.max_food
        max_food_level = self.max_player_level * len(self.players)
        # min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players)
        # max_obs = [field_x-1, field_y-1, max_food_level] * max_food + [
        #     field_x-1,
        #     field_y-1,
        #     self.max_player_level,
        # ] * len(self.players)

        min_obs = [-1, -1] * max_food + [-1, -1] * len(self.players)
        max_obs = [field_x-1, field_y-1] * max_food + [
            field_x-1,
            field_y-1,
        ] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 0 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 0 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def adjacent_player_number(self, row, col):
        return len([
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ])

    def spawn_players(self, max_player_level):
        for player in self.players:

            attempts = 0
            player.reward = 0
            player.additional_reward = 0

            while attempts < 1000:
                row = self.np_random.randint(self.rows // 2 - 1, self.rows // 2 + 1)
                col = self.np_random.randint(self.rows // 2 - 1, self.rows // 2 + 1)
                # if self._is_empty_location(row, col):
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        1,
                        self.field_size,
                    )
                    break
                attempts += 1

    def spawn_food(self, max_food, max_level):
        self.field[0            , 0]             = 2
        self.field[0            , self.cols - 1] = 2
        self.field[self.rows - 1, 0 ]            = 2
        self.field[self.rows - 1, self.cols - 1] = 2
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        # elif action == Action.LOAD:
        #     return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player, sight):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    reward=a.reward if a == player else None,
                    additional_reward=a.additional_reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, sight, a.position
                    )
                )
                <= 2 * sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, sight)),
            game_over=self.game_over,
            sight=sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, sight):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            # for i in range(self.max_food):
            #     obs[3 * i] = -1
            #     obs[3 * i + 1] = -1
            #     obs[3 * i + 2] = 0

            # for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
            #     obs[3 * i] = y
            #     obs[3 * i + 1] = x
            #     obs[3 * i + 2] = observation.field[y, x]

            # for i in range(len(self.players)):
            #     obs[self.max_food * 3 + 3 * i] = -1
            #     obs[self.max_food * 3 + 3 * i + 1] = -1
            #     obs[self.max_food * 3 + 3 * i + 2] = 0

            # for i, p in enumerate(seen_players):
            #     obs[self.max_food * 3 + 3 * i] = p.position[0]
            #     obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
            #     obs[self.max_food * 3 + 3 * i + 2] = p.level

            for i in range(self.max_food):
                obs[2 * i] = -1
                obs[2 * i + 1] = -1

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[2 * i] = y
                obs[2 * i + 1] = x

            for i in range(len(self.players)):
                obs[self.max_food * 2 + 2 * i] = -1
                obs[self.max_food * 2 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[self.max_food * 2 + 2 * i] = p.position[0]
                obs[self.max_food * 2 + 2 * i + 1] = p.position[1]

            return obs

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        def get_player_additional_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.additional_reward

        observations = [self._make_obs(player, sight=sight) for player in self.players]
        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        ninfo = {'additional_reward': get_player_additional_reward(observations[0])}
        if self.return_all_llm_rewards:
            ninfo['all_llm_rewards'] = self.return_reward_lst
        return nobs, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])
        
        self.collected_food = ""
        
        for i, player in enumerate(self.players):
            self.agents_position[str(i + 1)] = np.array(player.position)

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        if self.return_all_llm_rewards:
            self.return_reward_lst = []
            for llm_reward in self.additional_reward_lst[5:]:
                self.return_reward_lst.append(0)

        nobs, _, _, _ = self._make_gym_obs(sight=self.sight)
        return nobs
    
    def agent_food_distance(self, agent_idx: str, food_idx: str):
        agent_pos = self.agents_position[agent_idx]
        food_pos = self.foods_position[food_idx]
        distance = np.linalg.norm(agent_pos - food_pos)
        return distance

    def step(self, actions):
        self.current_step += 1
        for p in self.players:
            p.reward = 0
            p.additional_reward = 0

        ## process actions
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]
        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        ## move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)
        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        food_loaded_level = None
        food_loaded_position = None
        
        for i, player in enumerate(self.players):
            self.agents_position[str(i + 1)] = np.array(player.position)
            
        for player in self.players:
            # find adjacent food
            if self.adjacent_food(*player.position) == 0:
                # no adjacent food
                continue
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]
            adj_players = self.adjacent_players(frow, fcol)
            adj_player_level = sum([a.level for a in adj_players])
            if adj_player_level < food:
                # failed to load
                continue
            # else the food was loaded and each player scores points
            food_loaded_level = food
            food_loaded_position = (frow, fcol)
            self.field[frow, fcol] = 0
            
            for food, (food_row, food_col) in self.foods_position.items():
                if food_row == frow and food_col == fcol:
                    self.collected_food = food
            
            break

        ## reward the agents
        # self.additional_reward
        for player in self.players:
            if food_loaded_level is not None:
                player.reward = 1 / self.n_agents
            player.additional_reward = self.compute_additional_reward()

            if self.return_all_llm_rewards:
                self.return_reward_lst = []
                for llm_reward in self.additional_reward_lst[5:]:
                    self.return_reward_lst.append(llm_reward())

        self._game_over = (
            self.field.sum() < 8 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()
        
        return self._make_gym_obs(sight=self.sight)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def no_additional_reward(self):
        return 0

    def prefer_food_A(self):
        if self.collected_food == 'A':
            return 2
        elif self.collected_food in ['B', 'C', 'D']:
            return -1
        else:
            return 0

    def prefer_food_B(self):
        if self.collected_food == 'B':
            return 2
        elif self.collected_food in ['A', 'C', 'D']:
            return -1
        else:
            return 0

    def prefer_food_C(self):
        if self.collected_food == 'C':
            return 2
        elif self.collected_food in ['B', 'A', 'D']:
            return -1
        else:
            return 0

    def prefer_food_D(self):
        if self.collected_food == 'D':
            return 2
        elif self.collected_food in ['B', 'C', 'A']:
            return -1
        else:
            return 0
        
        
    '''
    2025-12-10_14-39-21
    Human players may prefer to collect the food that has the smallest total distance from both agents to it.
    '''
    def llm1(self) -> float:
        agent1_pos = self.agents_position["1"]
        agent2_pos = self.agents_position["2"]
        total_distances = []
        for food_pos in self.foods_position.values():
            total_distances.append(np.linalg.norm(agent1_pos - food_pos) + np.linalg.norm(agent2_pos - food_pos))
        return 0.1 / (min(total_distances) + 1e-6)

