# CODE FROM: https://github.com/alxdrcirilo/hex

from pygame import gfxdraw, font
import pygame
from math import cos, sin, pi, radians

from configparser import ConfigParser


class UI:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read("config.ini")

        self.board_size = self.config.getint("game", "board_size")
        self.hex_lookup = {}

        self.red = (222, 29, 47)
        self.blue = (0, 121, 251)
        self.green = (0, 255, 0)
        self.white = (255, 255, 255)
        self.black = (40, 40, 40)
        self.gray = (70, 70, 70)
        self.hex_radius = 60

        self.color_mapping = {
            (0, 0): self.white,  # Empty cell
            (0, 1): self.red,  # Player 1
            (1, 0): self.blue  # Player 2
        }

        self.color = {i: self.white for i in range(self.board_size ** 2)}

        self.x_offset, self.y_offset = 60, 60
        self.text_offset = 45
        self.screen = pygame.display.set_mode(
            (self.x_offset + (2 * self.hex_radius) * self.board_size + self.hex_radius * self.board_size,
             round(self.y_offset + (1.75 * self.hex_radius) * self.board_size)))
        pygame.font.init()
        self.fonts = font.SysFont("Sans", 20)

        self.screen.fill(self.black)

        self.hex_lookup = {}
        self.rects, self.color, self.node = [], [self.white] * (self.board_size ** 2), None

    def draw_board(self, hex_game):
        # Create a temporary surface to draw on
        temp_surface = pygame.Surface(self.screen.get_size())

        # Draw the board on the temporary surface
        counter = 0
        for row in range(self.board_size):
            for column in range(self.board_size):
                player_color = hex_game.grid[row][column]
                self.color[counter] = self.color_mapping[player_color]
                self.draw_hexagon(temp_surface, self.black, self.get_coordinates(row, column), counter)
                counter += 1
        self.draw_text()

        # Rotate the temporary surface 45 degrees
        rotated_surface = pygame.transform.rotate(temp_surface, 30)

        # Calculate the offset to center the rotated surface
        offset_x = (self.screen.get_width() - rotated_surface.get_width()) // 2 + 60
        offset_y = (self.screen.get_height() - rotated_surface.get_height()) // 2 - 60

        # Blit the rotated surface onto the main screen
        self.screen.fill(self.black)
        self.screen.blit(rotated_surface, (offset_x, offset_y))
        pygame.display.flip()

    def draw_text(self):
        alphabet = list(map(chr, range(97, 123)))

        for _ in range(self.board_size):
            # Columns
            text = self.fonts.render(alphabet[_].upper(), True, self.white, self.black)
            text_rect = text.get_rect()
            text_rect.center = (self.x_offset + (2 * self.hex_radius) * _, self.text_offset / 2)
            rotated_text = pygame.transform.rotate(text, 45)
            rotated_text_rect = rotated_text.get_rect(center=text_rect.center)
            self.screen.blit(rotated_text, rotated_text_rect)

            # Rows
            text = self.fonts.render(str(_), True, self.white, self.black)
            text_rect = text.get_rect()
            text_rect.center = (
            self.text_offset / 4 + self.hex_radius * _, self.y_offset + (1.75 * self.hex_radius) * _)
            rotated_text = pygame.transform.rotate(text, 45)
            rotated_text_rect = rotated_text.get_rect(center=text_rect.center)
            self.screen.blit(rotated_text, rotated_text_rect)

    def get_coordinates(self, row: int, column: int):
        x = self.x_offset + (2 * self.hex_radius) * column + self.hex_radius * row
        y = self.y_offset + (1.75 * self.hex_radius) * row

        # Rotate 45 degrees
        x_rotated = x - self.hex_radius / 2
        y_rotated = y + self.hex_radius / 2

        return x_rotated, y_rotated

    def draw_hexagon(self, surface: object, color: tuple, position: tuple, node: int):
        # Vertex count and radius
        n = 6
        x, y = position
        offset = 3

        # Outline
        self.hex_lookup[node] = [(x + (self.hex_radius + offset) * cos(radians(90) + 2 * pi * _ / n),
                                  y + (self.hex_radius + offset) * sin(radians(90) + 2 * pi * _ / n))
                                 for _ in range(n)]
        gfxdraw.aapolygon(surface,
                          self.hex_lookup[node],
                          color)

        # Shape
        gfxdraw.filled_polygon(surface,
                               [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
                                 y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
                                for _ in range(n)],
                               self.color[node])

        # Antialiased shape outline
        gfxdraw.aapolygon(surface,
                          [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
                            y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
                           for _ in range(n)],
                          self.black)

        # Placeholder
        rect = pygame.draw.rect(surface,
                                self.color[node],
                                pygame.Rect(x - self.hex_radius + offset, y - (self.hex_radius / 2),
                                            (self.hex_radius * 2) - (2 * offset), self.hex_radius))
        self.rects.append(rect)

        # Bounding box (colour-coded)
        bbox_offset = [0, 3]

        # Top side
        if 0 < node < self.board_size:
            points = ([self.hex_lookup[node - 1][3][_] - bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node - 1][4][_] - bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node][3][_] - bbox_offset[_] for _ in range(2)])
            gfxdraw.filled_polygon(surface,
                                   points,
                                   self.blue)
            gfxdraw.aapolygon(surface,
                              points,
                              self.blue)

        # Bottom side
        if self.board_size ** 2 - self.board_size < node < self.board_size ** 2:
            points = ([self.hex_lookup[node - 1][0][_] + bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node - 1][5][_] + bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node][0][_] + bbox_offset[_] for _ in range(2)])
            gfxdraw.filled_polygon(surface,
                                   points,
                                   self.blue)
            gfxdraw.aapolygon(surface,
                              points,
                              self.blue)

        # Left side
        bbox_offset = [3, -3]

        if node % self.board_size == 0:
            if node >= self.board_size:
                points = ([self.hex_lookup[node - self.board_size][1][_] - bbox_offset[_] for _ in range(2)],
                          [self.hex_lookup[node - self.board_size][0][_] - bbox_offset[_] for _ in range(2)],
                          [self.hex_lookup[node][1][_] - bbox_offset[_] for _ in range(2)])
                gfxdraw.filled_polygon(surface,
                                       points,
                                       self.red)
                gfxdraw.aapolygon(surface,
                                  points,
                                  self.red)

        # Right side
        if (node + 1) % self.board_size == 0:
            if node > self.board_size:
                points = ([self.hex_lookup[node - self.board_size][4][_] + bbox_offset[_] for _ in
                           range(2)],
                          [self.hex_lookup[node - self.board_size][5][_] + bbox_offset[_] for _ in
                           range(2)],
                          [self.hex_lookup[node][4][_] + bbox_offset[_] for _ in range(2)])
                gfxdraw.filled_polygon(surface,
                                       points,
                                       self.red)
                gfxdraw.aapolygon(surface,
                                  points,
                                  self.red)
