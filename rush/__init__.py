"""rush — a 2D team shooter for 500 v 500 battles with control points, and the soldier network that plays it.

    from rush.battle import play
    play("hf:koskokos/rush-soldier", "random", map_name="Front50", out="battle.arena.bin.gz")
"""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")     # the map code keeps pygame.Rect from the first versions
__version__ = "0.1.0"
