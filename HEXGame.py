from collections import deque

import numpy as np
import torch


class Hex:
    def __init__(self, width, height, current_player=(0, 1), last_action=None, grid=None):
        self.width = width
        self.height = height
        self.grid = grid if grid is not None else [[(0, 0) for _ in range(height)] for _ in range(width)]
        self.current_player = current_player
        self.last_action = last_action

    def get_possible_actions(self):
        """
        Get a list of all possible actions, including illegal moves.

        Returns:
        list: A list of all possible actions.
        """
        return [(x, y) for x in range(self.width) for y in range(self.height) if self.grid[x][y] == (0, 0)]

    def get_all_actions(self):
        """
        Get a list of all possible actions, including illegal moves.

        Returns:
            list: A list of all possible actions.
        """
        all_actions = [(row, col) for row in range(self.width) for col in range(self.height)]
        return all_actions

    def take_action(self, action):
        """
        Take an action on the current board state and return the resulting new state.

        Args:
            action (tuple): A (x, y) coordinate representing the action to take.

        Returns:
            Hex: A new Hex instance representing the resulting state after taking the action.

        Raises:
            ValueError: If the specified action is not a valid move (i.e., the cell is already occupied).
        """
        new_grid = [row[:] for row in self.grid]
        x, y = action
        if new_grid[x][y] != (0, 0):
            raise ValueError(f'Grid location ({x}, {y}) is already taken!')
        new_grid[x][y] = self.current_player
        new_player = (0, 1) if self.current_player == (1, 0) else (1, 0)
        new_state = Hex(self.width, self.height, new_player, action, new_grid)
        return new_state

    def get_action(self):
        """
        Returns the action taken to reach the current state.
        """
        return self.last_action

    def is_terminal(self):
        """
        Check if the current state is a terminal state (i.e., the game is over).

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.get_result((1, 0)) or self.get_result((0, 1))

    def get_result(self, player):
        """
        Check if the given player has a winning path on the current board state.

        Args:
            player (tuple): A (0, 1) or (1, 0) tuple representing the player to check for a win.

        Returns:
            bool: True if the player has a winning path, False otherwise.
        """
        visited = set()
        queue = deque()

        start_edge = [(x, 0) for x in range(self.width)] if player == (0, 1) else [(0, y) for y in range(self.height)]
        for cell in start_edge:
            if self.grid[cell[0]][cell[1]] == player:
                queue.append((cell, [cell]))

        while queue:
            (x, y), current_path = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Check if the current cell reaches the opposite edge
            if player == (0, 1) and y == self.height - 1 or player == (1, 0) and x == self.width - 1:
                self.winning_path = current_path  # Store the winning path
                return True

            # Check neighboring cells for the same player
            for dx, dy in [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[nx][ny] == player and (nx, ny) not in visited:
                    queue.append(((nx, ny), current_path + [(nx, ny)]))

        self.winning_path = []  # Clear path if no win
        return False

    def get_state_tensor(self):
        """
        Get the current board state as a tensor representation.

        Returns:
            torch.Tensor: A 4D tensor representing the current board state.
        """
        state_tensor = np.zeros((4, self.width, self.height), dtype=np.float32)

        # Planes 0, 1, 2: Black, White, Empty
        for x in range(self.width):
            for y in range(self.height):
                player = self.grid[x][y]
                if player == (0, 1):
                    state_tensor[0, x, y] = 1
                elif player == (1, 0):
                    state_tensor[1, x, y] = 1
                else:
                    state_tensor[2, x, y] = 1

        # Plane 3: Current player
        state_tensor[3] = 1 if self.current_player == 1 else 0

        return torch.from_numpy(state_tensor)

    @classmethod
    def from_state(cls, state):
        """
        Create a Hex instance from a given state tensor.

        Args:
            state (torch.Tensor): A 4D tensor representing the board state.

        Returns:
            Hex: A Hex instance representing the given state.
        """
        width = 7
        height = 7
        current_player = (0, 1) if state[0] == 1 else (1, 0)
        grid = [[0 for _ in range(height)] for _ in range(width)]

        for x in range(width):
            for y in range(height):
                grid[x][y] = (state[1 + x * height + y], 0)

        return cls(width, height, current_player=current_player, grid=grid)