Tello Waypoint Navigation
==========================

The Tello Waypoint Navigation provides waypoint-based navigation for DJI Tello drones with waypoint mapping, interactive navigation, direct waypoint navigation, sequential waypoint navigation and 360-degree surrounding scan as its main capabilities. Its basic capabilities include returning the drone instance that is currently in use by the Navigation Engine, drone take off and drone landing. 

Installation
-------------

To install Tello Waypoint Navigation run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[NAVIGATION_TELLO]

Usage
-------------

The Tello Waypoint Navigation module can have the following optional configurations to function:

#. NAVIGATION_TELLO_WAYPOINT_DIR - Directory path for storing waypoint files (default: default directory in user home folder)
#. NAVIGATION_TELLO_WAYPOINT_FILE - Specific waypoint file to use for navigation (default: None, decided later by the user or system)
#. NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED - Movement speed during mapping in cm/s (default: 38)
#. NAVIGATION_TELLO_MAPPING_ROTATION_SPEED - Rotation speed during mapping in degrees/s (default: 70)
#. NAVIGATION_TELLO_NAVIGATION_SPEED - Movement speed during navigation in cm/s (default: 55)
#. NAVIGATION_TELLO_VERTICAL_FACTOR - Vertical movement scaling factor (default: 1.0)
#. NAVIGATION_TELLO_IMAGE_DIR - Directory for scan operation images (default: default directory in user home folder)

Code Example
-------------

**Waypoint Mapping:**

.. code-block:: python

    from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
    from dronebuddylib.atoms.navigation import NavigationInstruction

    # Basic navigation engine setup
    engine_configs = EngineConfigurations({})

    # Optional configurations
    engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR, "/path/to/waypoints/directory")
    engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, 50)
    engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED, 70)

    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)

    # Waypoint mapping
    results = engine.map_location()

**Interactive Navigation:**

.. code-block:: python

    from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
    from dronebuddylib.atoms.navigation import NavigationInstruction

    # Basic navigation engine setup
    engine_configs = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)

    # Interactive navigation
    results = engine.navigate()

**Direct Waypoint Navigation:**

.. code-block:: python

    from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
    from dronebuddylib.atoms.navigation import NavigationInstruction

    # Basic navigation engine setup
    engine_configs = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)

    # Direct waypoint navigation
    result1 = engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    result2 = engine.navigate_to_waypoint("WP_001", NavigationInstruction.HALT)

**Sequential Waypoint Navigation:**

.. code-block:: python

    from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
    from dronebuddylib.atoms.navigation import NavigationInstruction

    # Basic navigation engine setup
    engine_configs = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)

    # Specify waypoint(s) for the drone to navigate to in a list 
    waypoints = ["WP_002", "WP_003", "Kitchen", "WP_001"]

    # Sequential waypoint navigation
    result = engine.navigate_to(waypoints, NavigationInstruction.HALT)

**360-Degree Surrounding Scan:**

.. code-block:: python

    from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
    from dronebuddylib.atoms.navigation import NavigationInstruction

    # Basic navigation engine setup
    engine_configs = EngineConfigurations({}) 
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)

    # 360-degree surrounding scan
    images = engine.scan_surrounding()

Output 
-------------

The output will be given in the following formats:

**Waypoint Mapping Results:**

.. code-block:: json

    [
        {"id": "WP_001", "name": "START"},
        {"id": "WP_002", "name": "Kitchen"},
        {"id": "WP_003", "name": "END"}
    ]

**Interactive Navigation Results:**

.. code-block:: json

    ["WP_002", "WP_003", "WP_001"]

**Direct Navigation Results:**

.. code-block:: json

    [False, "WP_002"]
    [True, "WP_001"]

**Sequential Waypoint Navigation Results:**

.. code-block:: json

    ["WP_002", "WP_003", "WP_001"]

**360-Degree Surrounding Scan Results:**

.. code-block:: json

    [
        {
            "image_path": "/path/to/image0.jpg",
            "filename": "image0.jpg",
            "waypoint_file": "waypoint_file.json",
            "waypoint": "WP_002",
            "rotation_from_start": 0,
            "image_number": 1,
            "timestamp": "20250805_143022_123",
            "format": "JPEG"
        },
        {
            "image_path": "/path/to/image1.jpg",
            "filename": "image1.jpg",
            "waypoint_file": "waypoint_file.json",
            "waypoint": "WP_002",
            "rotation_from_start": 15,
            "image_number": 2,
            "timestamp": "20250805_143023_456",
            "format": "JPEG"
        }
    ]