__version__ = "1.0.6"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from .follow_me import follow_me

from .follow import init_follower
from .follow import follow

from .get_pointed_obj import get_pointed_obj

from .hand_following import init_handFollower
from .hand_following import close_to_hand
