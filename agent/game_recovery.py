"""Bounded standard-library implementation for a well-specified car game."""

import re
import textwrap


def supports_car_game_recovery(task):
    text = str(task or "").lower()
    is_car_game = re.search(r"\b(car|racing|race)\b", text) and re.search(r"\b(game|racing|race)\b", text)
    requests_python_gui = re.search(r"\b(python|gui|tkinter|window)\b", text)
    return bool(is_car_game and requests_python_gui)


def car_game_files():
    return {
        "game_logic.py": textwrap.dedent('''\
            """Pure racing-game state and rules; safe to import without a GUI."""

            from dataclasses import dataclass


            LANES = 3
            ROAD_LENGTH = 12


            @dataclass
            class GameState:
                player_lane: int = 1
                score: int = 0
                distance: int = 0
                obstacle: tuple[int, int] | None = None
                game_over: bool = False


            def move_player(state: GameState, direction: int) -> GameState:
                """Move one lane left (-1) or right (+1), within track bounds."""
                if not state.game_over and direction in (-1, 1):
                    state.player_lane = max(0, min(LANES - 1, state.player_lane + direction))
                return state


            def advance(state: GameState, spawn_lane: int | None = None) -> GameState:
                """Advance traffic and distance by one frame and detect collisions."""
                if state.game_over:
                    return state
                state.distance += 1
                if state.obstacle is not None:
                    lane, row = state.obstacle
                    row += 1
                    if row >= ROAD_LENGTH - 1:
                        if lane == state.player_lane:
                            state.game_over = True
                        else:
                            state.score += 1
                        state.obstacle = None
                    else:
                        state.obstacle = (lane, row)
                if state.obstacle is None and spawn_lane in range(LANES):
                    state.obstacle = (spawn_lane, 0)
                return state


            def new_game() -> GameState:
                return GameState()
        '''),
        "main.py": textwrap.dedent('''\
            """Tkinter car-racing game. Run with: python main.py"""

            import random
            import tkinter as tk

            from game_logic import LANES, ROAD_LENGTH, GameState, advance, move_player, new_game


            WIDTH, HEIGHT = 360, 520
            ROAD_LEFT, ROAD_RIGHT = 54, 306
            CELL_WIDTH = (ROAD_RIGHT - ROAD_LEFT) / LANES
            CELL_HEIGHT = (HEIGHT - 100) / ROAD_LENGTH


            class CarRacingGame:
                def __init__(self, root):
                    self.root = root
                    self.root.title("Pocket Racer")
                    self.root.resizable(False, False)
                    self.state = new_game()
                    self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#17351f", highlightthickness=0)
                    self.canvas.pack()
                    # Keyboard input: arrows/A-D switch lanes; R restarts after a crash.
                    self.root.bind("<Left>", lambda _event: self.change_lane(-1))
                    self.root.bind("<Right>", lambda _event: self.change_lane(1))
                    self.root.bind("a", lambda _event: self.change_lane(-1))
                    self.root.bind("d", lambda _event: self.change_lane(1))
                    self.root.bind("r", lambda _event: self.restart())
                    self.draw()
                    self.root.after(180, self.update_game)

                def change_lane(self, direction):
                    """Apply keyboard input to move the player between road lanes."""
                    move_player(self.state, direction)
                    self.draw()

                def restart(self):
                    self.state = new_game()
                    self.draw()
                    self.root.after(180, self.update_game)

                def update_game(self):
                    if self.state.game_over:
                        return
                    spawn_lane = None
                    if self.state.obstacle is None and random.random() < 0.24:
                        spawn_lane = random.randrange(LANES)
                    advance(self.state, spawn_lane)
                    self.draw()
                    if not self.state.game_over:
                        self.root.after(180, self.update_game)

                def draw(self):
                    self.canvas.delete("all")
                    self.canvas.create_rectangle(ROAD_LEFT, 0, ROAD_RIGHT, HEIGHT, fill="#34383e", outline="")
                    for lane in range(1, LANES):
                        x = ROAD_LEFT + lane * CELL_WIDTH
                        for y in range(0, HEIGHT, 42):
                            self.canvas.create_line(x, y, x, y + 22, fill="#d9d9d9", width=2)
                    self.canvas.create_text(12, 22, anchor="w", fill="white", font=("Segoe UI", 13, "bold"),
                                            text=f"Distance {self.state.distance}   Passed {self.state.score}")
                    player_x = ROAD_LEFT + self.state.player_lane * CELL_WIDTH + CELL_WIDTH / 2
                    player_y = HEIGHT - 54
                    self.draw_car(player_x, player_y, "#35a7ff")
                    if self.state.obstacle is not None:
                        lane, row = self.state.obstacle
                        obstacle_x = ROAD_LEFT + lane * CELL_WIDTH + CELL_WIDTH / 2
                        obstacle_y = 65 + row * CELL_HEIGHT
                        self.draw_car(obstacle_x, obstacle_y, "#f05b52")
                    if self.state.game_over:
                        self.canvas.create_rectangle(35, 205, WIDTH - 35, 310, fill="#111820", outline="#ffffff")
                        self.canvas.create_text(WIDTH / 2, 235, fill="white", font=("Segoe UI", 20, "bold"),
                                                text="CRASH!")
                        self.canvas.create_text(WIDTH / 2, 275, fill="white", font=("Segoe UI", 12),
                                                text=f"Passed {self.state.score} cars  ·  Press R to restart")

                def draw_car(self, x, y, color):
                    self.canvas.create_rectangle(x - 19, y - 28, x + 19, y + 28,
                                                 fill=color, outline="white", width=2)
                    self.canvas.create_rectangle(x - 12, y - 17, x + 12, y - 4,
                                                 fill="#c9efff", outline="")
                    self.canvas.create_oval(x - 22, y - 18, x - 16, y - 7, fill="#17191c", outline="")
                    self.canvas.create_oval(x + 16, y - 18, x + 22, y - 7, fill="#17191c", outline="")
                    self.canvas.create_oval(x - 22, y + 9, x - 16, y + 20, fill="#17191c", outline="")
                    self.canvas.create_oval(x + 16, y + 9, x + 22, y + 20, fill="#17191c", outline="")


            def main():
                root = tk.Tk()
                CarRacingGame(root)
                root.mainloop()


            if __name__ == "__main__":
                main()
        '''),
        "README.md": textwrap.dedent('''\
            # Pocket Racer

            A small, playable top-down car-racing game built with Python's standard-library Tkinter.
            No images, downloads, or third-party Python packages are needed.

            ## Run

            ```powershell
            python main.py
            ```

            Use **Left/Right arrows** or **A/D** to change lanes. Avoid the red traffic cars.
            Distance advances while you drive; passed cars increase your score. After a crash,
            press **R** to restart.

            ## Tests

            ```powershell
            python -m unittest discover -s tests -v
            ```

            The game rules are isolated in `game_logic.py`, so tests do not open a GUI window.
        '''),
        "requirements.txt": "# No external packages required; Tkinter is included with standard Python installations.\n",
        "tests/test_game_logic.py": textwrap.dedent('''\
            import unittest

            from game_logic import LANES, ROAD_LENGTH, advance, move_player, new_game


            class GameLogicTests(unittest.TestCase):
                def test_player_movement_stays_inside_the_track(self):
                    state = new_game()
                    move_player(state, -1)
                    move_player(state, -1)
                    self.assertEqual(state.player_lane, 0)
                    move_player(state, 1)
                    self.assertEqual(state.player_lane, 1)
                    self.assertEqual(LANES, 3)

                def test_distance_and_score_advance_when_traffic_is_passed(self):
                    state = new_game()
                    state.obstacle = (0, ROAD_LENGTH - 2)
                    advance(state)
                    self.assertEqual(state.distance, 1)
                    self.assertEqual(state.score, 1)
                    self.assertFalse(state.game_over)

                def test_collision_stops_the_game(self):
                    state = new_game()
                    state.obstacle = (state.player_lane, ROAD_LENGTH - 2)
                    advance(state)
                    self.assertTrue(state.game_over)
                    self.assertEqual(state.score, 0)
                    self.assertEqual(advance(state).distance, 1)

                def test_new_game_restarts_score_and_state(self):
                    state = new_game()
                    state.distance = 50
                    state.score = 3
                    state.game_over = True
                    restarted = new_game()
                    self.assertEqual((restarted.distance, restarted.score), (0, 0))
                    self.assertFalse(restarted.game_over)
                    self.assertEqual(restarted.player_lane, 1)


            if __name__ == "__main__":
                unittest.main()
        '''),
    }
