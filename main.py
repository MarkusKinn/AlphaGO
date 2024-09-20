from GameManager import GameManager

from HEXGame import Hex
from ANET import ANET
from configparser import ConfigParser

weights_pretrained = ['models_pretrained/weights_episode_0.pt', 'models_pretrained/weights_episode_1.pt',
                      'models_pretrained/weights_episode_2.pt', 'models_pretrained/weights_episode_3.pt']

weights = ['models/weights_generation_0.pt', 'models/weights_generation_1.pt',
           'models/weights_generation_2.pt', 'models/weights_generation_3.pt']

config = ConfigParser()
config.read("config.ini")

game_manager = GameManager()
# game_manager.run_rl_training()
game_manager.run_tournament(weights, config.getint("game", "num_topp_games"))

from ActorClient import ActorClient

actor = ActorClient()
class Client(ActorClient):
    def __init__(self):
        super().__init__(qualify=True, auth='0f09d6f13f2e46fab9f04007bc0dc810')
        self.hex_game = None
        self.current_model = ANET(board_size=7)
        self.current_model.load_weights('models/weights_episode_1000_best.pt')

    def handle_game_start(self, start_player):
        self.logger.info('Game start: start_player=%s', start_player)
        self.hex_game = Hex(7, 7)
        self.hex_game.current_player = (0, 1) if start_player == 1 else (1, 0)

    def handle_get_action(self, state):
        self.hex_game = self.hex_game.from_state(state)

        possible_actions = self.hex_game.get_possible_actions()
        state_tensor = self.hex_game.get_state_tensor()
        best_action, _ = self.current_model.predict(state_tensor.unsqueeze(0), possible_actions,
                                                    stochastic=False)
        self.logger.info('Get action: state=%s', state)
        row, col = best_action
        self.logger.info('Picked random: row=%s col=%s', row, col)
        return row, col

# client = Client()
# client.run()
