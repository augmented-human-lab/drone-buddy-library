__version__ = "1.0.6"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from .follow_me import follow_me

from .follower_engine import init_follower
from .follower_engine import follow

from .object_pointer_engine import get_pointed_obj

from .hand_follower_engine import init_handFollower
from .hand_follower_engine import fix_target_to_hand

from .hand_following import init_handFollower
from .hand_following import close_to_hand

from .fly_around import fly_around
from .fly_around import init_flyArounder