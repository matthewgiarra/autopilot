
import CoDrone
import pygame
import time
from pdb import set_trace

PAD_DOWN=(0,-1)
PAD_UP=(0,1)
PAD_LEFT=(-1,0)
PAD_RIGHT=(1,0)

class fakeDrone():
    def __init__(self):
        self.connected = False
        self.flying = False
    def pair(self, drone_id = None):
        self.connected = True
    def disconnect(self):
        self.connected = False
    def isConnected(self):
        return self.connected
    def is_flying(self):
        return self.flying
    def takeoff(self):
        self.flying = True
    def land(self):
        self.flying = False
    def emergency_stop(self):
        self.flying = False
        self.connected = False
    def move(self, roll, pitch, yaw, throttle):
        return

def get_controls(joystick, sensitivity=100):
    # Sensitivity: 0-100, adjusts full range of controls
    # Now get the stick states
    roll  = sensitivity * joystick.get_axis(2)
    pitch = -1 * sensitivity * joystick.get_axis(3)
    yaw = sensitivity * joystick.get_axis(0)
    throttle = -1 * sensitivity * joystick.get_axis(1)
    return roll, pitch, yaw, throttle

def startPressed(joystick, button_id = 9):
    return (joystick.get_button(button_id) == 1)
def selectPressed(joystick, button_id = 8):
    return (joystick.get_button(button_id) == 1)
def startSelectPressed(joystick, start_button = 9, select_button = 8):
    return (startPressed(joystick, start_button) and selectPressed(joystick, select_button))

# ID of this drone
drone_id = 8898

# Fake drone?
drone_is_fake = False

# Instantiate drone object
if drone_is_fake:
    drone = fakeDrone()
else:
    drone = CoDrone.CoDrone()

# Initialize pygame
pygame.init()
pygame.joystick.init()
print("Detected " + str(pygame.joystick.get_count()) + " joystick")
joystick = pygame.joystick.Joystick(0)

# Pressing "start" and "select" together pairs the drone
print("Hello, CoDrone")
print("Press start to pair.")

try:
    done = False
    # paired = False
    # while paired is False:
    # while not drone.isConnected():
    while not drone.isConnected():
        for event in pygame.event.get(): # User did something.
            if event.type == pygame.QUIT: # If user clicked close.
                done = True # Flag that we are done so we exit this loop.
            elif event.type == pygame.JOYBUTTONDOWN and startPressed(joystick):
                print("Pairing...")
                drone.pair(str(drone_id))

    print("Drone paired.")
    print("Controls:\n\tTake off: D-pad up\n\tLand: D-pad down\n\tQuit: start + sel")

    # while drone.isConnected(): # Replace with while drone.isConnected()
    while drone.isConnected():
        for event in pygame.event.get(): # User did something.
            if event.type == pygame.QUIT: # If user clicked close.
                done = True # Flag that we are done so we exit this loop.
            elif event.type == pygame.JOYBUTTONDOWN and startSelectPressed(joystick):
                if drone.is_flying():
                        print("Emergency stopping...")
                        drone.emergency_stop()
                print("Unpairing CoDrone...")
                drone.disconnect()
                print("Drone unpaired.")
            elif event.type == pygame.JOYHATMOTION:
                hat = joystick.get_hat(0)
                if hat == PAD_DOWN:
                    if drone.is_flying():
                        print("Landing...")
                        drone.land()
                elif hat == PAD_UP:
                    if not drone.is_flying():
                        print("Taking off...")
                        drone.takeoff()

            # Now get the stick states
            roll, pitch, yaw, throttle = get_controls(joystick)
            if drone.is_flying():
                drone.move(roll, pitch, yaw, throttle)
                print("[r,p,y,t] = %0.1f, %0.1f, %0.1f, %0.1f" % (roll, pitch, yaw, throttle))

except Exception as e:
    print("Exception raised: " + str(e))

finally:
    if drone.is_flying():
        print("Emergency stopping...")
        drone.emergency_stop()
        
    # Disconnect from the drone
    if drone.isConnected():
        print("Unpairing CoDrone...")
        drone.disconnect()
        print("Unpaired CoDrone.")
    print("Goodbye.")


