Supported models
==========

Tello
~~~~~~~~~~~~~~~~~~~~~~~~~

The DJI Tello is a compact programmable drone designed for education and development. DroneBuddy integrates with Tello drones through the Tello SDK protocol, which enables programmatic control via WiFi communication over UDP sockets.

Please refer to the `DJI Tello SDK 2.0 User Guide <https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf>`_ and `djitellopy Python SDK documentation <https://djitellopy.readthedocs.io/en/latest/>`_ for more details about the underlying drone platform and communication protocols.

Here's a simplified explanation of how Tello drone communication and control works:

#. **WiFi Connection and SDK Mode**: The Tello drone creates its own WiFi access point (typically named "TELLO-XXXXXX") that client devices connect to. Once connected, the drone must be placed in SDK mode by sending the "command" instruction to establish programmatic control. The drone communicates over specific UDP ports: 8889 for commands, 8890 for state information, and 11111 for video streaming.

#. **UDP Command Protocol**: All drone control is achieved through UDP text commands sent to the drone's IP address (192.168.10.1:8889). Commands include basic flight operations (takeoff, land, up, down, left, right, forward, back, cw, ccw), advanced movements (go x y z speed, curve x1 y1 z1 x2 y2 z2 speed), and status queries (battery?, height?, attitude?, speed?). Each command returns "ok" for success or "error" for failure.

#. **Real-time Telemetry and State Monitoring**: The Tello continuously broadcasts state information on port 8890, including attitude data (pitch, roll, yaw angles), flight status, battery percentage, motor temperatures, and altitude readings. DroneBuddy's navigation system actively monitors these telemetry streams to track drone orientation, battery levels, and flight status for safety and navigation accuracy.

#. **Movement Execution and Timing**: When executing navigation commands, the Tello processes movement instructions sequentially. Movement commands like "forward 50" instruct the drone to move 50 centimeters forward at default speed. The navigation system calculates movement durations based on commanded distance and speed, adding timing compensation for acceleration and deceleration phases to ensure accurate distance tracking.

#. **Orientation Control and Yaw Management**: The Tello maintains its orientation through yaw angle control using clockwise (cw) and counter-clockwise (ccw) rotation commands. DroneBuddy's navigation system queries the drone's current yaw angle using "attitude?" commands and calculates the required rotation to achieve the target heading before executing linear movements. Yaw angles are normalized to the -180 to +180 degree range for consistent navigation.

#. **Video Streaming Integration**: The Tello provides real-time video streaming over UDP port 11111, which DroneBuddy integrates during manual mapping operations. The video stream uses H.264 encoding and requires proper frame handling and display threading. This live video feed assists operators during manual waypoint mapping by providing visual feedback of the drone's perspective.

#. **Safety and Error Handling**: The Tello SDK includes built-in safety features such as automatic landing on low battery, and emergency stop capabilities. DroneBuddy extends these safety features with additional battery monitoring threads, emergency shutdown protocols, and graceful connection management to ensure safe operation during autonomous navigation missions.

Tello's key advantages for navigation applications include its stability, precise movement execution, comprehensive telemetry feedback, and reliable WiFi communication protocol. The drone's compact size and indoor flight capabilities make it ideal for waypoint navigation in confined environments where larger drones cannot operate effectively.

DroneBuddy's Tello integration supports the full range of SDK capabilities including:

- **Basic Movement Commands**: takeoff, land, up, down, left, right, forward, back, cw, ccw
- **Advanced Navigation**: go x y z speed for direct coordinate movement
- **Status Monitoring**: battery, height, attitude, speed, and temperature queries
- **Emergency Controls**: Emergency stop and immediate landing capabilities
- **Video Streaming**: Real-time H.264 video stream integration for mapping operations

The Tello communication protocol is designed for reliability with command acknowledgment, timeout handling, and automatic retry mechanisms. Commands are processed sequentially to ensure precise movement execution, and the drone maintains flight stability through its internal flight controller while executing navigation instructions.

Tello drones have evolved through multiple generations (Tello, Tello EDU) with enhanced SDK capabilities, improved flight stability, and extended battery life. The SDK protocol has remained consistent across generations, ensuring compatibility with DroneBuddy's navigation implementation.

The navigation system leverages Tello's precise movement capabilities and real-time telemetry to achieve centimeter-level navigation accuracy. By combining the drone's built-in flight stability with DroneBuddy's waypoint recording and pathfinding algorithms, the system provides reliable autonomous navigation for indoor environments.

Keep in mind that while the Tello provides excellent programmability and control precision, it has limitations including indoor-only operation (no GPS), limited payload capacity, and dependence on WiFi connectivity. The drone's flight time is approximately 13 minutes, requiring careful battery monitoring during extended navigation missions.

Important Considerations
------------------------

While the Tello drone offers excellent programmability and SDK integration, it's important to note that its performance depends on WiFi signal strength, environmental conditions, and proper SDK command timing. The drone is designed for indoor operation and educational use. Regular firmware updates and proper connection management are necessary to ensure the Tello operates effectively within DroneBuddy's navigation system.
