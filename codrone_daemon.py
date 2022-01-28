
import CoDrone
import numpy as np
import pygame
import time
import inspect
from pdb import set_trace

# ID of this drone
drone_id = 8898

# Fake drone?
drone_is_fake = False

# Some constants
PAD_DOWN=(0,-1)
PAD_UP=(0,1)
PAD_LEFT=(-1,0)
PAD_RIGHT=(1,0)

# trim = trim + (stick_position - bias)
# input =  trim + (stick_position - bias)

class controlInput():
    def __init__(self,roll=0,pitch=0,yaw=0,throttle=0):
        self.roll     = roll
        self.pitch    = pitch
        self.yaw      = yaw
        self.throttle = throttle

    def __str__(self):
        return("[%0.1f, %0.1f, %0.1f, %0.1f]" % 
        (self.roll, self.pitch, self.yaw, self.throttle))

class controlTrim():
    def __init__(self,roll=0,pitch=0,yaw=0,throttle=0):
        self.roll     = roll
        self.pitch    = pitch
        self.yaw      = yaw
        self.throttle = throttle

    def set(self, roll=0, pitch=0, yaw=0,throttle=0):
        self.roll     = roll     
        self.pitch    = pitch    
        self.yaw      = yaw      
        self.throttle = throttle 

    def update(self, roll=0,pitch=0,yaw=0,throttle=0):
        self.roll     += roll # Add biases
        self.pitch    += pitch
        self.yaw      += yaw
        self.throttle += throttle

    def zero(self):
        self.set(0,0,0,0)
    
    def __str__(self):
        return("[%0.1f, %0.1f, %0.1f, %0.1f]" % 
        (self.roll, self.pitch, self.yaw, self.throttle))

def print_function_name():
    fun_name = inspect.stack()[1][3]
    print("<" + fun_name + ">")

class fakeDrone():
    def __init__(self):
        self.connected = False
        self.flying = False

    def isConnected(self):
        return self.connected
    def is_flying(self):
        return self.flying
    
    def pair(self, drone_id = None):
        self.connected = True
        print_function_name()
    def disconnect(self):
        self.connected = False
        print_function_name()
    def takeoff(self):
        self.flying = True
        print_function_name()
    def land(self):
        self.flying = False
        print_function_name()
    def emergency_stop(self):
        self.flying = False
        print_function_name()
    def move(self, roll, pitch, yaw, throttle):
        print_function_name()
    def arm_pattern(self, Color=None, Mode=None, Speed=None):
        print_function_name()
    def reset_default_led(self):
        print_function_name()

def get_controls(joystick, stick_sensitivity=100, scale_throttle = False, bias = controlInput()):
    # Sensitivity: 0-100, adjusts full range of controls
    # Get the stick states
    roll  = stick_sensitivity * (joystick.get_axis(2) - bias.roll)
    pitch = -1 * stick_sensitivity * (joystick.get_axis(3) - bias.pitch)
    yaw = stick_sensitivity * (joystick.get_axis(0) - bias.yaw)
    
    # Don't mess with the throttle because a "low sensitivity"
    # can cause the drone to descend, versus just slowing the response 
    if scale_throttle is True:
        throttle_sens = stick_sensitivity
    else:
        throttle_sens = 100
    throttle = -1 * throttle_sens * (joystick.get_axis(1) - bias.throttle)
    return roll, pitch, yaw, throttle

# Apply trim to the control input
def trim_controls(trim = controlTrim(0,0,0,0), roll=0,pitch=0,yaw=0,throttle=0, minval=-100, maxval=100):
    roll = np.clip(roll + trim.roll, minval, maxval)
    pitch = np.clip(pitch + trim.pitch, minval, maxval)
    yaw = np.clip(yaw + trim.yaw, minval, maxval)
    throttle = np.clip(throttle + trim.throttle, minval, maxval)
    return roll, pitch, yaw, throttle

def startPressed(joystick, button_id = 9):
    return (joystick.get_button(button_id) == 1)
def selectPressed(joystick, button_id = 8):
    return (joystick.get_button(button_id) == 1)
def startSelectPressed(joystick, start_button = 9, select_button = 8):
    return (startPressed(joystick, start_button) and selectPressed(joystick, select_button))

def LB_PRESSED(joystick, button_id = 4):
    return (joystick.get_button(button_id) == 1)
def RB_PRESSED(joystick, button_id = 5):
    return (joystick.get_button(button_id) == 1)
def LT_PRESSED(joystick, button_id = 6):
    return (joystick.get_button(button_id) == 1)
def RT_PRESSED(joystick, button_id = 7):
    return (joystick.get_button(button_id) == 1)
def A_PRESSED(joystick, button_id = 1):
    return (joystick.get_button(button_id) == 1)
def Y_PRESSED(joystick, button_id = 3):
    return (joystick.get_button(button_id) == 1)

# Initialize some toggle states
lb_down = False
lt_down = False
a_button_down = False
y_button_down = False

# Initialize stick sensitivity (min 0, max 100)
stick_sensitivity = 100
stick_sens_step = 10
stick_sens_min = 10 # If this goes negative your controls will filp
stick_sens_max = 100

# Stick trims
trim = controlTrim()

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

# Get the stick bias
biasRoll,biasPitch,biasYaw,biasThrottle = get_controls(joystick, stick_sensitivity=1, scale_throttle = True)
biasRoll = joystick.get_axis(2)
biasPitch = joystick.get_axis(3)
biasYaw = joystick.get_axis(0)
biasThrottle = joystick.get_axis(1)
stickBias = controlInput(roll=biasRoll, pitch=biasPitch, yaw=biasYaw,throttle=biasThrottle)
print("Bias: " + str(stickBias))


try:
    done = False
    
    # Connect to the drone
    while not drone.isConnected():
        for event in pygame.event.get(): # User did something.
            if event.type == pygame.QUIT: # If user clicked close.
                done = True # Flag that we are done so we exit this loop.
            elif event.type == pygame.JOYBUTTONDOWN and startPressed(joystick):
                print("Pairing...")
                drone.pair(str(drone_id))

    print("Drone paired.")
    print("Controls:")
    print("\tTake off: D-pad up")
    print("\tLand: D-pad down")
    print("\tEmergency stop: start")
    print("\tQuit: start + select")
    
    # Initialize stick sensitivity so user can't screw themselves up too badly with initial settings
    stick_sensitivity = np.clip(stick_sensitivity, 10, 100)

    # while drone.isConnected(): # Replace with while drone.isConnected()
    while drone.isConnected():
        for event in pygame.event.get(): # User did something.
            
            # User quits. Doesn't do anything for now.
            if event.type == pygame.QUIT: # If user clicked close.
                done = True # Flag that we are done so we exit this loop.
            
            # When buttons are depressed
            elif event.type == pygame.JOYBUTTONDOWN:
               
                # Start button pressed: stop the drone if it's flying
                if startPressed(joystick):
                    if drone.is_flying():
                        print("Emergency stopping...")
                        drone.emergency_stop()
                    # Start and select pressed: disconnect
                    if selectPressed(joystick):
                        print("Unpairing CoDrone...")
                        drone.disconnect()
                        print("Drone unpaired")
                
                # Left button down: Flash LEDs to indicate trimming
                if LB_PRESSED(joystick):
                    drone.arm_pattern(CoDrone.Color.Blue, CoDrone.Mode.BLINK, 25)
                    # TODO: Update joystick trim info so that we can fine-tune trimming
                    lb_down = not lb_down
                
                # Left trigger down: reset trim
                if LT_PRESSED(joystick):
                    if lt_down is False:
                        drone.arm_pattern(CoDrone.Color.White, CoDrone.Mode.BLINK, 25)
                        trim.zero()
                        drone.reset_default_led()
                        lt_down = not lt_down # Toggle the LT
                        # print(trim)
                        print("Reset trim: [r,p,y,t] = " + str(trim))
                
                # "A" button down (button 1): decrease stick sensitivity
                if A_PRESSED(joystick):
                    if a_button_down is False:
                        stick_sensitivity = np.clip(stick_sensitivity - stick_sens_step, stick_sens_min, stick_sens_max)
                        a_button_down = not a_button_down
                        print("Stick sensitivity %0d%%" % stick_sensitivity)

                # "Y" button down (button 3): increase stick sensitivity
                if Y_PRESSED(joystick):
                    if y_button_down is False:
                        stick_sensitivity = np.clip(stick_sensitivity + stick_sens_step, stick_sens_min, stick_sens_max)
                        y_button_down = not y_button_down
                        print("Stick sensitivity %0d%%" % stick_sensitivity)

            # When buttons are released
            elif event.type == pygame.JOYBUTTONUP:

                # Left button up 
                # Set the trim to the stick positions when LB is released
                if lb_down is True:
                    
                    # Get trim values
                    roll, pitch, yaw, throttle = get_controls(joystick, stick_sensitivity, bias=stickBias)
                    
                    # Set trim
                    trim.update(roll=roll,pitch=pitch,yaw=yaw,throttle=throttle)
                    drone.reset_default_led()
                    lb_down = not lb_down # Toggle the LB
                    # print(trim)
                    print("Set trim: [r,p,y,t] = " + str(trim))

                # Left trigger up 
                if lt_down is True:
                    lt_down = not lt_down # Toggle the LT

                # A button up
                if a_button_down is True:
                    a_button_down = not a_button_down # Toggle the button

                # Y button up
                if y_button_down is True:
                    y_button_down = not y_button_down # Toggle the button

            # D-Pad input (take off, land)                
            elif event.type == pygame.JOYHATMOTION:

                # Get D-pad state
                hat = joystick.get_hat(0) 

                # Take off
                if hat == PAD_UP:
                    if not drone.is_flying():
                        print("Taking off...")
                        drone.takeoff()
                        drone.move(0,0,0,0) # Zero out controls
                # Land
                elif hat == PAD_DOWN:
                    if drone.is_flying():
                        print("Landing...")
                        drone.land()
            
            # Command drone based on joystick input
            elif event.type == pygame.JOYAXISMOTION:
                # Get the stick states
                roll, pitch, yaw, throttle = get_controls(joystick, 
                    stick_sensitivity = stick_sensitivity, bias=stickBias)
                roll, pitch, yaw, throttle = trim_controls(trim=trim, roll=roll, pitch=pitch,yaw=yaw,throttle=throttle)
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
        drone.reset_default_led()
        print("Unpairing CoDrone...")
        drone.disconnect()
        print("Unpaired CoDrone.")
    print("Goodbye.")


