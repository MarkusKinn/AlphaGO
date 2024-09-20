import time
from collections import deque

from pytorch_lightning import Trainer
import pygame

from HEXGame import Hex
from RBUF import Replay_Buffer
from UI import UI
from MCTS.MCTS import MCTSSearch
from ANET import ANET
from configparser import ConfigParser


class RLSystem:
    def __init__(self):
        # Initialize the system with configuration from a file
        self.config = ConfigParser()
        self.config.read("config.ini")

        # Initialize game and UI components
        self.hex_game = None
        self.board_size = self.config.getint("game", "board_size")
        self.display = self.config.getboolean("game", "display")
        self.episodes = self.config.getint("game", "episodes")
        self.epochs = self.config.getint("game", "epochs")
        self.models = self.config.getint("game", "models")
        self.mcts_iterations = self.config.getint("game", "iterations")
        self.cooldown = self.config.getfloat("game", "cooldown")

        # Initialize the replay buffer and UI
        self.replay_buffer = Replay_Buffer(self.config.getint("game", "buffer"))
        self.ui = UI()

    def run_and_train(self):
        """
        Run the RLSystem and train the model.
        """
        # Initialize Pygame if display is enabled
        if self.display:
            pygame.init()
            pygame.display.set_caption('Hex Game')

        # Initialize the ANET model and save the initial weights
        model = ANET(self.config)
        model.save_weights(f"./models/weights_generation_0.pt")

        # Train the model for multiple intervals
        for interval in range(self.models):
            # Collect data for multiple episodes
            for episode in range(self.episodes):
                # Initialize a new Hex game
                self.hex_game = Hex(self.board_size, self.board_size)

                # Play the game until it's terminal
                while not self.hex_game.is_terminal():
                    # Handle Pygame events if display is enabled
                    if self.display:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return

                    # Perform MCTS search to select the best action
                    mcts = MCTSSearch(self.hex_game)
                    best_action = mcts.search_anet_rollout(self.mcts_iterations, model)

                    # Collect training data from the MCTS search
                    training_data = mcts.collect_training_data()
                    self.replay_buffer.add(training_data)
                    self.replay_buffer.training_data.extend(training_data)

                    # Take the best action in the game
                    self.hex_game = self.hex_game.take_action(best_action)
                    time.sleep(self.cooldown)

                    # Update the UI if display is enabled
                    if self.display:
                        self.ui.draw_board(hex_game=self.hex_game)
                        pygame.display.flip()

            # Train the model on the collected data
            model.training_data = self.replay_buffer.get_training_data()
            train_loader = model.train_dataloader()
            self.trainer = Trainer(max_epochs=self.epochs)
            self.trainer.fit(model, train_loader)
            model.save_weights(f"./models/weights_generation_{interval + 1}.pt")
            self.replay_buffer.clear()

        # Quit Pygame if display is enabled
        if self.display:
            pygame.quit()

    def train(self, sizes):
        full_replay_buffer = Replay_Buffer.load_from_disk('5x5_1500epi1500.pkl')
        full_training_data = full_replay_buffer.get_training_data()
        i = 0
        for size in sizes:
            self.replay_buffer = Replay_Buffer(size)
            self.replay_buffer.buffer = deque(full_training_data[-size:], maxlen=size)
            model = ANET(self.config)
            model.training_data = self.replay_buffer.get_training_data()
            train_loader = model.train_dataloader()
            self.trainer = Trainer(max_epochs=self.epochs)
            self.trainer.fit(model, train_loader)
            model.save_weights(f"./models_pretrained/weights_episode_{i}.pt")
            self.replay_buffer.clear()
            i += 1