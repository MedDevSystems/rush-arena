from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Spatial; CONCEPT(9): GridAccelerated; TECH(7): math+set]
## @modulecontract
## @purpose Spatial utility functions with grid-accelerated wall lookups. Line-of-sight and raycasting check a boolean grid instead of iterating all wall rects — O(ray_length) instead of O(ray_length × n_walls).
## @scope Pure geometric queries — no game logic, no entity mutation.
## @input Positions (Vector2 or tuples), wall grid (built once from rects)
## @output Boolean visibility, distance lists
## @links USES_API(6): pygame.Rect; USES_API(5): math; LINKS_TO: config
## @invariants
## - WallGrid is built once and reused for all queries
## - Functions are side-effect-free (given the same grid)
## @rationale
## Q: Why a boolean grid instead of spatial hash?
## A: Tile-aligned walls map perfectly to a grid. One lookup per ray step (O(1)) vs iterating 376 rects. 100x+ speedup on 60x60 map.
## @changes
## LAST_CHANGE: [v0.2.0] Grid-accelerated lookups for 60x60 arena performance.
## @modulemap
## CLASS 8[Boolean grid of wall tiles for O(1) lookup] => WallGrid
## FUNC 8[Check if two points can see each other] => has_line_of_sight
## FUNC 8[Cast N evenly spaced rays] => cast_wall_rays
## @usecases
## - [WallGrid]: load_map -> build WallGrid -> pass to all spatial queries
## - [has_line_of_sight]: Bot/Env -> VisibilityCheck -> bool
## - [cast_wall_rays]: Env -> BuildObservation -> WallDistances
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: line of sight, raycast, visibility, walls, spatial, grid, O(1) lookup, performance
# STRUCTURE: ▶ WallGrid(walls) → bool[rows][cols] → ⚡ has_LOS(a,b) steps along ray, grid[ty][tx] → ◇ hit ? → ⎋

import math

import pygame

from rush.config import ARENA_COLS, ARENA_H, ARENA_ROWS, ARENA_W, TILE_SIZE


# region CLASS_WallGrid [DOMAIN(8): Spatial; CONCEPT(9): GridAccelerated; TECH(7): list]
## @purpose Boolean 2D grid where True means "this tile is a wall". Built once from wall rects for O(1) collision checks.
## @uses TILE_SIZE, ARENA_COLS, ARENA_ROWS
## @complexity 3
class WallGrid:

    def __init__(self, walls: list[pygame.Rect], cols: int = ARENA_COLS, rows: int = ARENA_ROWS, tile_size: int = TILE_SIZE) -> None:
        self._cols = cols
        self._rows = rows
        self._tile_size = tile_size
        self.grid: list[list[bool]] = [
            [False] * cols for _ in range(rows)
        ]
        for w in walls:
            col = w.x // tile_size
            row = w.y // tile_size
            if 0 <= row < rows and 0 <= col < cols:
                self.grid[row][col] = True

    def safe_spawn_tiles(self) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for r in range(1, self._rows - 1):
            for c in range(1, self._cols - 1):
                if self.grid[r][c]:
                    continue
                clear = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if self.grid[r + dr][c + dc]:
                            clear = False
                            break
                    if not clear:
                        break
                if clear:
                    result.append((r, c))
        return result

    # region FUNC_rect_hits_wall
    ## @purpose Rect-vs-walls test through the tile index instead of scanning every wall.
    ## @io (x, y, w, h) -> bool
    ## @rationale
    ## Q: Why is this not just any(rect.colliderect(w) for w in walls)?
    ## A: That scan is O(walls) — 1311 rects on the Arena map — and it runs twice per agent
    ## A: per frame for movement plus once per bullet. Every wall is exactly one tile, so the
    ## A: only candidates are the tiles the rect overlaps: at most four for a player.
    ## A: Same answer, ~300x fewer comparisons.
    def rect_hits_wall(self, x: float, y: float, w: float, h: float) -> bool:
        ts = self._tile_size
        # pygame.Rect truncates to integers; mirror that so the answer stays identical.
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + int(w) - 1, y0 + int(h) - 1
        # Outside the arena there are no wall rects at all, so a scan would answer False
        # there. Clamping (instead of reporting a hit) keeps this equivalent: anything
        # reaching inward still meets the border tiles, which are walls.
        row0, row1 = max(0, y0 // ts), min(self._rows - 1, y1 // ts)
        col0, col1 = max(0, x0 // ts), min(self._cols - 1, x1 // ts)
        for row in range(row0, row1 + 1):
            grid_row = self.grid[row]
            for col in range(col0, col1 + 1):
                if grid_row[col]:
                    return True
        return False
    # endregion FUNC_rect_hits_wall

    def is_wall(self, px: float, py: float) -> bool:
        col = int(px) // self._tile_size
        row = int(py) // self._tile_size
        if col < 0 or col >= self._cols or row < 0 or row >= self._rows:
            return True
        return self.grid[row][col]

# endregion CLASS_WallGrid


# region FUNC_has_line_of_sight [DOMAIN(8): Spatial; CONCEPT(8): Visibility; TECH(6): math]
## @purpose March a ray from point A to point B; return False if any wall tile is hit. Uses WallGrid for O(1) per step.
## @uses WallGrid.is_wall
## @io Vector2, Vector2, WallGrid -> bool
## @complexity 4
def has_line_of_sight(
    a: pygame.Vector2,
    b: pygame.Vector2,
    walls: list[pygame.Rect],
    step: float = 8.0,
    wall_grid: WallGrid | None = None,
) -> bool:
    dx = b.x - a.x
    dy = b.y - a.y
    dist = math.hypot(dx, dy)
    if dist < step:
        return True
    n_steps = int(dist / step)
    sx = dx / n_steps
    sy = dy / n_steps
    cx, cy = a.x, a.y

    if wall_grid is not None:
        for _ in range(n_steps):
            cx += sx
            cy += sy
            if wall_grid.is_wall(cx, cy):
                return False
    else:
        for _ in range(n_steps):
            cx += sx
            cy += sy
            for w in walls:
                if w.collidepoint(cx, cy):
                    return False
    return True
# endregion FUNC_has_line_of_sight


# region FUNC_cast_wall_rays [DOMAIN(7): Spatial; CONCEPT(8): Raycasting; TECH(6): math]
## @purpose Cast N evenly spaced rays from a position, return normalized distances. Uses WallGrid for O(1) per step.
## @uses WallGrid.is_wall
## @io Vector2, list[pygame.Rect], int, float -> list[float]
## @complexity 5
def cast_wall_rays(
    pos: pygame.Vector2,
    walls: list[pygame.Rect],
    n_rays: int,
    max_dist: float,
    step: float = 8.0,
    wall_grid: WallGrid | None = None,
    arena_w: int = ARENA_W,
    arena_h: int = ARENA_H,
) -> list[float]:
    results: list[float] = []
    for i in range(n_rays):
        angle = (2 * math.pi * i) / n_rays
        dx_step = math.cos(angle) * step
        dy_step = math.sin(angle) * step
        dist = 0.0
        cx, cy = pos.x, pos.y
        hit = False
        while dist < max_dist:
            cx += dx_step
            cy += dy_step
            dist += step
            if cx < 0 or cx >= arena_w or cy < 0 or cy >= arena_h:
                hit = True
                break
            if wall_grid is not None:
                if wall_grid.is_wall(cx, cy):
                    hit = True
                    break
            else:
                for w in walls:
                    if w.collidepoint(cx, cy):
                        hit = True
                        break
                if hit:
                    break
        results.append(dist / max_dist)
    return results
# endregion FUNC_cast_wall_rays
