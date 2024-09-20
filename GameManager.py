from RLSystem import RLSystem
from TournamentSystem import TournamentSystem

class GameManager:
    def __init__(self):
        self.rl_system = RLSystem()
        self.tournament_system = TournamentSystem()

    def run_rl_training(self):
        self.rl_system.run_and_train()

    def run_tournament(self, model_weights_paths, num_games):
        self.tournament_system.tournament(model_weights_paths, num_games)