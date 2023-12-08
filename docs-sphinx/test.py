import os
import sys

try:
    import dronebuddylib
    print("im working")
except ImportError:
    print("oh no 1")

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "dronebuddylib")))

try:
    import dronebuddylib
except ImportError:
    print("oh no 2")

# import deeplearning  # does not fail

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import dronebuddylib  # does not fail anymore
