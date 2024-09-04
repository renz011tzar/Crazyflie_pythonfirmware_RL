import time
import cflib.crtp  # Crazyflie library
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.utils import uri_helper

# Initialize the low-level drivers
cflib.crtp.init_drivers(enable_debug_driver=True)

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M')

def simple_flight_sequence(scf):
    """
    Function to perform a simple flight sequence using the HighLevelCommander.
    """
    try:
        # Create a HighLevelCommander instance to send high-level commands
        commander = HighLevelCommander(scf.cf)
        
        print("Taking off...")
        commander.takeoff(0.5, 2.0)  # Take off to 0.5 meters height over 2 seconds
        time.sleep(2.5)  # Wait for the takeoff to complete

        print("Hovering...")
        time.sleep(3)  # Hover for 3 seconds

        # Move in a square pattern
        print("Moving in a square pattern...")
        square_size = 0.5  # 0.5 meters

        # Move to the right
        commander.go_to(square_size, 0.0, 0.5, 0, 2.0)  # Move 0.5m to the right over 2 seconds
        time.sleep(2.5)

        # Move forward
        commander.go_to(square_size, square_size, 0.5, 0, 2.0)  # Move 0.5m forward over 2 seconds
        time.sleep(2.5)

        # Move to the left
        commander.go_to(0.0, square_size, 0.5, 0, 2.0)  # Move 0.5m to the left over 2 seconds
        time.sleep(2.5)

        # Move back to starting position
        commander.go_to(0.0, 0.0, 0.5, 0, 2.0)  # Move back to the starting position over 2 seconds
        time.sleep(2.5)

        print("Landing...")
        commander.land(0.0, 2.0)  # Land slowly over 2 seconds
        time.sleep(2.5)  # Wait for the landing to complete

        print("Flight sequence complete. Stopping commander.")
        commander.stop()
    except Exception as e:
        print("Error during flight sequence: ", str(e))

# Connect to the Crazyflie and run the flight sequence
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
    simple_flight_sequence(scf)
