import time

import pygame

from HEXGame import Hex
from RBUF import Replay_Buffer
from UI import UI
from ANET import ANET
from configparser import ConfigParser


class TournamentSystem:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read("config.ini")
        self.hex_game = None
        self.board_size = self.config.getint("game", "board_size")
        self.replay_buffer = Replay_Buffer(self.config.getint("game", "buffer"))
        self.display = self.config.getboolean("game", "display")
        self.ui = UI()
        self.cooldown = self.config.getfloat("game", "cooldown")


    def play_game(self, model1, model2):
        """
        Play a single game between two models.

        Args:
            model1 (ANET): The first model.
            model2 (ANET): The second model.

        Returns:
            The winner of the game (1 if model1 wins, 2 if model2 wins).
        """
        game = Hex(self.board_size, self.board_size)
        current_player = 1

        if self.display:
            pygame.init()
            pygame.display.set_caption('Hex Game')

        while not game.is_terminal():
            if current_player == 1:
                possible_actions = game.get_possible_actions()
                state_tensor = game.get_state_tensor()
                best_action, _ = model1.predict(state_tensor.unsqueeze(0), possible_actions,
                                                stochastic=False)
            else:
                possible_actions = game.get_possible_actions()
                state_tensor = game.get_state_tensor()
                best_action, _ = model2.predict(state_tensor.unsqueeze(0), possible_actions,
                                                stochastic=False)

            game = game.take_action(best_action)
            time.sleep(self.cooldown)

            current_player = 3 - current_player

            if self.display:
                self.ui.draw_board(hex_game=game)
                pygame.display.update()
                # pygame.time.wait(100)

        if game.get_result((0, 1)):
            return 1
        else:
            return 2

    def tournament(self, model_weights_paths, num_games):
        """
        Run a tournament between multiple models.

        Args:
            model_weights_paths (list of str): List of paths to model weights files.
            num_games (int): Number of games to play between each pair of models.

        Returns:
            A dictionary with the win-loss records for each model.
        """

        models = []
        for path in model_weights_paths:
            model = ANET(self.config)
            model.load_weights(path)
            models.append(model)

        results = {f"Model {i + 1}": {"wins": 0, "losses": 0} for i in range(len(models))}

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                print(f"Model {i + 1} plays against Model {j + 1}")
                for _ in range(num_games // 2):
                    # Model i plays as player 1
                    winner = self.play_game(models[i], models[j])
                    if winner == 1:
                        results[f"Model {i + 1}"]["wins"] += 1
                        results[f"Model {j + 1}"]["losses"] += 1
                    else:
                        results[f"Model {i + 1}"]["losses"] += 1
                        results[f"Model {j + 1}"]["wins"] += 1

                for _ in range(num_games - num_games // 2):
                    # Model j plays as player 1
                    winner = self.play_game(models[j], models[i])
                    if winner == 1:
                        results[f"Model {j + 1}"]["wins"] += 1
                        results[f"Model {i + 1}"]["losses"] += 1
                    else:
                        results[f"Model {j + 1}"]["losses"] += 1
                        results[f"Model {i + 1}"]["wins"] += 1

        print("Final stats:")
        for model, stats in results.items():
            print(f"{model}: {stats['wins']} wins, {stats['losses']} losses")

        return results
