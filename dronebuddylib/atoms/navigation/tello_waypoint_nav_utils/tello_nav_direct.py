from djitellopy import Tello
from .tello_waypoint_nav_coordinator import TelloWaypointNavCoordinator

class TelloNavDirect:
    def __init__(self):
        if TelloWaypointNavCoordinator._active_instance is None: 
            self.tello = None
        else: 
            self.tello = TelloWaypointNavCoordinator._active_instance.tello

    def forward(self, distance: float) -> bool:
        """
        Moves the Tello drone forward by a specified distance.

        Args:
            distance (float): The distance to move forward in centimeters.
        """
        if self.tello is not None: 
            return self.move_forward(distance)
        else:
            #TODO: Takeoff, move, land
            pass
    
    def move_forward(self, distance: float) -> bool:
        """
        Moves the Tello drone forward by a specified distance in a blocking manner.

        Args:
            distance (float): The distance to move forward in centimeters.
        """ 
        try: 
            self.tello.move_forward(distance)
            return True
        except Exception as e:
            print(f"Error occurred while moving forward: {e}")
            return False