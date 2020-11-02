import collections
import math
import typing
from typing import Dict,List,Optional, NamedTuple


### HELPERS ###
MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object) :
    """
    a class that holds the min max vals of the tree
    """

    def __init__(self,known_bounds: Optional[KnownBounds]):
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE

    def update(self,value:float):
        self.minimum = min(self.minimum,value)
        self.maximum = max(self.maximum,value)

    def normalize(self,value:float) -> float:
        if self.maximum > self.minimum :
            # normalize only when we have set max and min vals
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MuZeroConfig(object) :
    def __init__(self,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None):
        """

        :param action_space_size:
        :param discount:
        :param dirichlet_alpha:
        :param num_simulations:
        :param td_steps:
        :param lr_init:
        :param lr_decay_steps:
        :param visit_softmax_temperature_fn:
        :param known_bounds:
        """
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1e6)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)

def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30 :
            return 1.
        else :
            return 0. # play according to the max

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1)
    )

def make_chess_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)

def make_go_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)

def make_shogi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)

def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)

class Action(object) :

    def __init__(self, index:int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class Player(object) :
    pass

class Node(object) :
    def __init__(self, prior:float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0 : return 0
        return self.value_sum / self.visit_count

class ActionHistory(object):
    """
    Simple history container used inside the search
    only used to keep track of actions executed
    """

    def __init__(self,history: List[Action], action_space_size:int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action:Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()

class Environment(object) :
    """the environment that RickZero is interacting with"""

    def step(self, action):
        pass

class Game(object):
    """A single episode of interaction with the environment"""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment() # game specific environment
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # game specific termination rules
        pass

    def legal_actions(self) -> List[Action]:
        # game specific calculations of legal actions
        return []

    def apply(self, action:Action):
        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root:Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index:int):
        # Game-specific feature planes
        return []

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        # the value target is the  discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1) :
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values) :
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else :
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else :
                # states past the end of games are treated as absorbing states
                targets.append((0,0,[]))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer(object):

    def __init__(self, config:MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps:int, td_steps:int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # sample game from buffer either uniformly or according to some priority
        return self.buffer[0]

    def sample_position(self, game) -> int:
        # sample position from game either uniformly or according to some priority
        return -1

class NetworkOutput(NamedTuple):
    value:float
    reward:float
    policy_logits:Dict[Action, float]
    hidden_state:List[float]

class Network(object):

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0,0,{},[])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0,0,{},[])

    def get_weights(self):
        # returns the weights of the network
        return []

    def training_steps(self) -> int:
        # how many steps / batches the network has been trained for
        return 0

class SharedStorage(object):

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else :
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step:int, network:Network):
        self._networks[step] = network

### END OF HELPERS ###
######################












# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    return 0, 0


def launch_job(f, *args):
    f(*args)
def make_uniform_network():
  return Network()