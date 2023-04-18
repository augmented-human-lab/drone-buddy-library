import unittest

from djitellopy import tello

import dronebuddylib.atoms as dbl_atoms
import dronebuddylib.molecules as dbl_molecules
from dronebuddylib import DroneCommands
from dronebuddylib.models import EngineBank


def drone_commands_executor(audio_command, helper_engine):
    voice_engine = helper_engine.get_text_to_speech_engine()
    drone = helper_engine.get_drone_instance()

    if audio_command == DroneCommands.TAKEOFF:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm taking off")
        drone.takeoff()

    if audio_command == DroneCommands.UP:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm going up")
        drone.send_rc_control(0, 0, 10, 0)
    if audio_command == DroneCommands.DOWN:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm going down")
        drone.send_rc_control(0, 0, -10, 0)
    if audio_command == DroneCommands.STOP:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm stopping")
        drone.send_rc_control(0, 0, 0, 0)
    if audio_command == DroneCommands.LEFT:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm moving to the left")
        drone.send_rc_control(-10, 0, 0, 0)
    if audio_command == DroneCommands.RIGHT:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm moving to the right")
        drone.send_rc_control(10, 0, 0, 0)
    if audio_command == DroneCommands.FORWARD:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm moving forward")
        drone.send_rc_control(0, -10, 0, 0)
    # if audio_command == DroneCommands.BATTERY:
    # battery_status = "Battery level is " + str(me.get_battery())
    # dbl_atoms.generate_speech_and_play(voice_engine, battery_status)
    if audio_command == DroneCommands.BACKWARD:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm moving backwards")
        drone.send_rc_control(0, 10, 0, 0)
    elif audio_command == DroneCommands.LAND:
        dbl_atoms.generate_speech_and_play(voice_engine, "I'm landing")
        drone.land()


class TestAudioProcessorPyaudio(unittest.TestCase):

    def test_upload_image(self):
        drone = tello.Tello()
        pyaudio_cred = dbl_molecules.init_pyaudio()

        helper_engine = EngineBank()
        helper_engine.set_object_detection_yolo_engine(
            r"C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\test\resources\objectdetection\yolov3.weights")
        helper_engine.set_speech_to_text_engine('en-us')
        helper_engine.set_drone_instance(drone)

        dbl_molecules.run_audio_recognition(pyaudio_cred, drone_commands_executor, helper_engine)


if __name__ == '__main__':
    unittest.main()
