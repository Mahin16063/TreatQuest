import os
import glob
import pygame
import shutil
from levels.levelAssets import Levels

# from q_table import QLearningAgent

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


class GridWorldEnv:
    TILE_SIZE = 64

    def __init__(self, level_files, asset_dir="assets"):
        pygame.mixer.init()
        self.level_files = level_files
        self.asset_dir = asset_dir
        self.current_level = 0
        self.grid = None
        self.pet_pos = None
        self.pet_surface = None
        self.tile_surfaces = {}
        self.objects = {}
        self.remaining_treats = 0
        self._load_assets()
        self.total_treats = 0
        self.collected_treats = 0
        self.step_count = 0
        self.sounds = {
            "treat": pygame.mixer.Sound("assets/sounds/treat.wav"),
            "trap": pygame.mixer.Sound("assets/sounds/trap.wav"),
            "level_complete": pygame.mixer.Sound("assets/sounds/level_complete.wav"),
            "background_music": pygame.mixer.Sound("assets/sounds/level_2.mp3"),
        }
        # path to temporary level file used for learning (copied from original on reset)
        self.temp_level_file = None

        for s in self.sounds.values():
            s.set_volume(0.6)

    # Function to get number of remaining treats
    def get_treat_count(self):
        return self.remaining_treats

    # Function to get total number of treats in level
    def get_total_treats(self):
        return self.total_treats

    # ----------------------------
    # Helper functions
    # ----------------------------
    def get_window_size(self):
        """Return (width, height) in pixels for the current grid"""
        rows = len(self.grid)
        cols = len(self.grid[0])
        return (cols * self.TILE_SIZE, rows * self.TILE_SIZE)

    # ----------------------------
    # Asset loading
    # ----------------------------
    def _safe_load(self, *parts):
        """Load and scale an image from the assets folder"""
        path = os.path.join(self.asset_dir, *parts)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing asset: {path}")
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, (self.TILE_SIZE, self.TILE_SIZE))

    def _load_assets(self):

        # default images (used if level not found in dict)
        default_tiles = {
            "T": "tile_1.png",  # treat
            "X": "tile_2.png",  # trap
            "#": "tile_3.png",  # wall
            ".": "tile_4.png",  # floor
        }

        # overrides the default tiles based on the current level
        if (self.current_level + 1) in Levels:
            level_pair = Levels[self.current_level + 1]
            default_tiles["."] = level_pair[0]
            default_tiles["#"] = level_pair[1]
            default_tiles["T"] = level_pair[2]
            default_tiles["X"] = level_pair[3]

        self.tile_surfaces = {
            "T": self._safe_load("tiles", default_tiles["T"]),
            "X": self._safe_load("tiles", default_tiles["X"]),
            "#": self._safe_load("tiles", default_tiles["#"]),
            ".": self._safe_load("tiles", default_tiles["."]),
            "P": None,  # Pet drawn separately
        }
        self.pet_surface = self._safe_load("pets", "siameseFront.png")

    # ----------------------------
    # Animations
    # ----------------------------

    def _load_trap_animation(self, tile=None):
        """
        Load trap frames from:
        assets/tiles/animated assets/level N/anime_*.png
        Falls back to the static trap tile if none found.
        """
        tile = tile or self.TILE_SIZE

        level_folder = f"level {self.current_level + 1}"
        folder = os.path.join(self.asset_dir, "tiles", "animated assets", level_folder)
        pattern = os.path.join(folder, "anime_*.png")

        self.trap_frames = []
        for path in sorted(glob.glob(pattern)):
            try:
                surf = pygame.image.load(path).convert_alpha()
                surf = pygame.transform.smoothscale(surf, (tile, tile))
                self.trap_frames.append(surf)
            except Exception as e:
                print(f"Exception: {e}")
                pass

        # Debug to see what it did
        print(
            f"[trap anim] level={self.current_level+1} folder={folder} frames={len(self.trap_frames)}"
        )

        if not self.trap_frames:
            # fall back to static trap tile if available
            static = self.tile_surfaces.get("X")
            self.trap_frames = [static] if static is not None else [None]

        self.trap_ms_per_frame = 500  # ~5 miliseconds wait per frame change

    # --------------------------------
    # Helper functions for animations
    # --------------------------------

    def _trap_current_frame(self):
        """Return the current frame for the animated trap."""
        if not getattr(self, "trap_frames", None):
            return None
        now = pygame.time.get_ticks()
        idx = (now // self.trap_ms_per_frame) % len(self.trap_frames)
        return self.trap_frames[idx]

    # ----------------------------
    # Map loading
    # ----------------------------
    def _load_map(self, filename):
        grid = []
        with open(filename, "r") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                grid.append(list(row))

        print(f"Loaded {filename}:")  # Debug
        for row in grid:
            print("".join(row))

        return grid

    def _generate_objects(self):
        self.objects = {}
        self.remaining_treats = 0

        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "P":
                    self.pet_pos = [r, c]
                elif ch == "T":
                    self.objects[(r, c)] = "T"
                    self.remaining_treats += 1
                elif ch == "X":
                    self.objects[(r, c)] = "X"
                elif ch == "#":
                    self.objects[(r, c)] = "#"

        # Store total treats at this level
        self.total_treats = self.remaining_treats

    # ----------------------------
    # Environment API
    # ----------------------------
    def reset(self, level_index=0):
        self.current_level = level_index

        # Create a temporary copy of the original level file that we'll use for learning.
        orig_path = self.level_files[level_index]
        levels_dir = os.path.dirname(orig_path) or "."
        temp_name = f"temp_{os.path.basename(orig_path)}"
        temp_path = os.path.join(levels_dir, temp_name)
        try:
            shutil.copyfile(orig_path, temp_path)
            self.temp_level_file = temp_path
            print(
                f"\nRESET: Copied original level file {orig_path} to temp file {temp_path}"
            )
        except Exception as e:
            print(
                f"Warning: could not create temp level file: {e}. Using original level file."
            )
            self.temp_level_file = orig_path

        self._load_assets()
        # Load from temp file so modifications are persistent for the learning run
        self.grid = self._load_map(self.temp_level_file)
        self._generate_objects()
        self._load_trap_animation(tile=self.TILE_SIZE)
        return self.grid

    def move_pet(self, action):
        dr, dc = 0, 0
        r, c = self.pet_pos
        if self.step_count % 2 == 0:
            num = "2"
        else:
            num = ""
        if action == "UP":
            self.pet_surface = self._safe_load("pets", f"siameseBack{num}.png")
            dr, dc = -1, 0
        elif action == "DOWN":
            self.pet_surface = self._safe_load("pets", f"siameseFront{num}.png")
            dr, dc = 1, 0
        elif action == "LEFT":
            self.pet_surface = self._safe_load("pets", f"siameseLeft{num}.png")
            dr, dc = 0, -1
        elif action == "RIGHT":
            self.pet_surface = self._safe_load("pets", f"siameseRight{num}.png")
            dr, dc = 0, 1

        nr, nc = self.pet_pos[0] + dr, self.pet_pos[1] + dc
        self.step_count += 1

        # Stay inside bounds
        if nr < 0 or nr >= len(self.grid) or nc < 0 or nc >= len(self.grid[0]):
            return False, "wall"

        # Check wall
        if self.objects.get((nr, nc)) == "#":
            return False, "wall"

        self.pet_pos = [nr, nc]

        # Check object interactions
        if (nr, nc) in self.objects:
            obj = self.objects.pop((nr, nc))

            if obj == "T":
                # Update both runtime objects and the grid file used for learning
                self.remaining_treats -= 1
                print(f"\nTREAT: Updating temp file {self.temp_level_file}")
                print("Grid before update:")
                for row in self.grid:
                    print("".join(row))

                try:
                    self.grid[nr][nc] = "."
                    self._write_temp_map()
                    print("\nGrid after update:")
                    for row in self.grid:
                        print("".join(row))
                except Exception as e:
                    print(f"Warning: failed to update temp level file: {e}")

                if "treat" in self.sounds:
                    self.sounds["treat"].play()
                if self.remaining_treats == 0:
                    print("Level complete!")
                    if "level_complete" in self.sounds:
                        self.sounds["level_complete"].play()
                    self._next_level()
                    return True, "finished"
                return False, "treat"

            elif obj == "X":
                print("Game over! Pet hit a trap.")
                if "trap" in self.sounds:
                    self.sounds["trap"].play()
                # Reset will copy original level back into the temp file
                self.reset(self.current_level)
                return False, "trap"

        return False, "empty"

    def _next_level(self):
        if self.current_level + 1 < len(self.level_files):
            self.reset(self.current_level + 1)
        else:
            print("You win! All levels complete.")

    # ----------------------------
    # Rendering
    # ----------------------------
    def render_console(self):
        for r, row in enumerate(self.grid):
            line = ""
            for c, ch in enumerate(row):
                if [r, c] == self.pet_pos:
                    line += "P"
                elif (r, c) in self.objects:
                    line += self.objects[(r, c)]
                else:
                    line += "."
            print(line)

    def render_pygame(self, screen):
        rows = len(self.grid)
        cols = len(self.grid[0])

        for r in range(rows):
            for c in range(cols):
                # Always draw background first
                bg_tile = self.tile_surfaces["."]
                screen.blit(bg_tile, (c * self.TILE_SIZE, r * self.TILE_SIZE))

                # Draw foreground objects if present
                if (r, c) in self.objects:
                    obj = self.objects[(r, c)]

                    # Animated trap: draw current frame instead of static tile
                    if obj == "X":
                        frame = self._trap_current_frame()
                        if frame is not None:
                            screen.blit(frame, (c * self.TILE_SIZE, r * self.TILE_SIZE))
                        else:
                            # fallback to default or just the static version of it
                            tile = self.tile_surfaces.get("X")
                            if tile:
                                screen.blit(
                                    tile, (c * self.TILE_SIZE, r * self.TILE_SIZE)
                                )
                    else:
                        tile = self.tile_surfaces.get(obj)
                        if tile:
                            screen.blit(tile, (c * self.TILE_SIZE, r * self.TILE_SIZE))

    def render_ui(self, screen):
        total_treats = self.get_total_treats()  # total in level
        collected = total_treats - self.remaining_treats  # already picked up

        # Apple icon
        apple_icon = pygame.transform.scale(self.tile_surfaces["T"], (32, 32))
        screen.blit(apple_icon, (10, 10))

        # Counter text
        font = pygame.font.Font(None, 36)
        counter_text = f"{collected}/{total_treats}"
        text_surface = font.render(counter_text, True, (255, 255, 255))
        screen.blit(text_surface, (50, 15))

        # Progress bar background
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 50
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))

        # Progress fill
        if total_treats > 0:
            progress = collected / total_treats
            fill_width = int(bar_width * progress)
            if fill_width > 0:
                if progress < 0.5:
                    color = (255, 100, 100)  # red
                elif progress < 1.0:
                    color = (255, 255, 100)  # yellow
                else:
                    color = (100, 255, 100)  # green

                pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))

        # Progress bar border
        pygame.draw.rect(
            screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2
        )

        # Completion message
        if collected == total_treats and total_treats > 0:
            complete_font = pygame.font.Font(None, 24)
            complete_text = complete_font.render(
                "All treats collected!", True, (100, 255, 100)
            )
            screen.blit(complete_text, (bar_x, bar_y + 25))

        # Draw pet
        screen.blit(
            self.pet_surface,
            (self.pet_pos[1] * self.TILE_SIZE, self.pet_pos[0] * self.TILE_SIZE),
        )

    def render_hud(self, screen, mode="PLAYER", episode=None, total_reward=None, epsilon=None):
        """
        Draws HUD on the top-right corner.
        Small text, clean layout, supports training and gameplay.
        """

        font = pygame.font.SysFont("Arial", 18)

        lines = [
            f"Level: {self.current_level + 1}",
            f"Steps: {self.step_count}",
            f"Mode: {mode}",
        ]

        if episode is not None:
            lines.append(f"Episode: {episode}")

        if total_reward is not None:
            lines.append(f"Reward: {total_reward}")

        if epsilon is not None:
            lines.append(f"Epsilon: {epsilon:.3f}")

        # Draw each line aligned to top-right
        y = 10
        for text in lines:
            surf = font.render(text, True, (255, 255, 255))
            x = screen.get_width() - surf.get_width() - 15
            screen.blit(surf, (x, y))
            y += 22  

    def get_state(self):
        """Return the current state as an integer index for Q-learning."""
        cols = len(self.grid[0])
        return self.pet_pos[0] * cols + self.pet_pos[1]

    def step(self, action_idx):
        """
        Take an action by index, return (next_state, reward, done, info)
        """
        action = ACTIONS[action_idx]
        level_changed, tile = self.move_pet(action)

        # Reward logic
        if tile == "finished":
            reward = 15
            done = True
        elif tile == "trap":
            reward = -50
            done = True
        elif tile == "treat":
            reward = 15
            done = False
        elif tile == "empty":
            reward = -1
            done = False
        elif tile == "wall":
            reward = -5
            done = False
        else:
            reward = 0
            done = False

        next_state = self.get_state()
        info = {"tile": tile}
        return next_state, reward, done, info

    @property
    def num_states(self):
        """Number of possible states (positions) in the grid."""
        return len(self.grid) * len(self.grid[0])

    @property
    def num_actions(self):
        """Number of possible actions."""
        return len(ACTIONS)

    # ----------------------------
    # File helpers for temp level persistence
    # ----------------------------
    def _write_temp_map(self):
        """Write current grid to the temp level file so collected treats are removed from disk used for learning."""
        if not self.temp_level_file:
            return
        try:
            with open(self.temp_level_file, "w", encoding="utf-8") as f:
                for row in self.grid:
                    f.write("".join(row) + "\n")
        except Exception as e:
            print(f"Error writing temp level file '{self.temp_level_file}': {e}")
