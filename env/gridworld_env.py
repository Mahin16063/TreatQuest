import os
import pygame

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class GridWorldEnv:
    TILE_SIZE = 64

    def __init__(self, level_files, asset_dir="assets"):
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
        self.tile_surfaces = {
            "T": self._safe_load("tiles", "tile_1.png"),
            "X": self._safe_load("tiles", "tile_2.png"),
            "#": self._safe_load("tiles", "tile_3.png"),
            ".": self._safe_load("tiles", "tile_4.png"),
            "P": None,  # Pet drawn separately
        }
        self.pet_surface = self._safe_load("pets", "orange-cat.png")

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

    # ----------------------------
    # Environment API
    # ----------------------------
    def reset(self, level_index=0):
        self.current_level = level_index
        self.grid = self._load_map(self.level_files[level_index])
        self._generate_objects()
        return self.grid

    def move_pet(self, action):
        dr, dc = 0, 0
        if action == "UP":
            dr, dc = -1, 0
        elif action == "DOWN":
            dr, dc = 1, 0
        elif action == "LEFT":
            dr, dc = 0, -1
        elif action == "RIGHT":
            dr, dc = 0, 1

        nr, nc = self.pet_pos[0] + dr, self.pet_pos[1] + dc

        # Stay inside bounds
        if nr < 0 or nr >= len(self.grid) or nc < 0 or nc >= len(self.grid[0]):
            return False  # no level change

        # Check wall
        if self.objects.get((nr, nc)) == "#":
            return False

        self.pet_pos = [nr, nc]

        # Check object interactions
        if (nr, nc) in self.objects:
            obj = self.objects.pop((nr, nc))
            if obj == "T":
                self.remaining_treats -= 1
                print("Picked up a treat!")
                if self.remaining_treats == 0:
                    print("Level complete!")
                    self._next_level()
                    return True  # signal level changed
            elif obj == "X":
                print("Game over! Pet hit a trap.")
                self.reset(self.current_level)
                return False

        return False  # no level change

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
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                tile = self.tile_surfaces.get(ch)

                if tile is None and ch != "P":
                    raise ValueError(
                        f"Unrecognized tile character '{ch}' at row {r}, col {c}"
                    )

                if tile:
                    screen.blit(tile, (c * self.TILE_SIZE, r * self.TILE_SIZE))

        # Draw objects
        for (r, c), obj in self.objects.items():
            tile = self.tile_surfaces.get(obj)
            if tile:
                screen.blit(tile, (c * self.TILE_SIZE, r * self.TILE_SIZE))

        # Draw pet
        screen.blit(
            self.pet_surface,
            (self.pet_pos[1] * self.TILE_SIZE, self.pet_pos[0] * self.TILE_SIZE),
        )
   