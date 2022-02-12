
from constants import CKeys, CColors
import CoDrone
from simple_pid import PID
import numpy as np
import cv2
import pygame
import streams
import inspect
import time
from pdb import set_trace

# ID of this drone
drone_id = 8898

# Fake drone?
drone_is_fake = False

# Threshold value of trace(kalman error covariance matrix) for signaling "tracking locked"
kf_err_cov_tracking_threshold = 2.5
autopilot_available = False # Autopilot unavailable to start
autopilot_armed = False
autopilot_enabled = False # Autopilot disabled to start

# Some constants
PAD_DOWN=(0,-1)
PAD_UP=(0,1)
PAD_LEFT=(-1,0)
PAD_RIGHT=(1,0)

# Target position in homogeneous camera coordinates (homogeneous coordinates, e.g., [x,y,z,1])
xyz_target = np.array([0,0,0.8,1], dtype=np.float32)

# Autopilot PID gains gains
# gains_roll = [80, 40, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [100, 0, 0] # Controls Y coordinate (along yaw axis)
# gains_pitch = [80, 40, 0] # controls Z coortinate (along roll axis)

# Autopilot PID gains gains
gains_roll = [80, 80, 10] # Controls X coordinate (along pitch axis)
gains_thrust = [100, 0, 1] # Controls Y coordinate (along yaw axis)
gains_pitch = [80, 80, 10] # controls Z coortinate (along roll axis)

# gains_roll = [20, 200, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [200, 180, 25] # Controls Y coordinate (along yaw axis)
# gains_pitch = [20, 200, 0] # controls Z coortinate (along roll axis)

### This kind of worked!!
# gains_roll = [100, 0, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [0, 0, 0] # Controls Y coordinate (along yaw axis)
# gains_pitch = [100, 0, 0] # controls Z coortinate (along roll axis)

# gains_roll = [500, 875, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [0, 0, 0] # Controls Y coordinate (along yaw axis)
# gains_pitch = [500, 875, 0] # controls Z coortinate (along roll axis)

# gains_roll = [500, 0, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [0, 0, 0] # Controls Y coordinate (along yaw axis)
# gains_pitch = [500, 0, 0] # controls Z coortinate (along roll axis)

# gains_roll = [0, 0, 0] # Controls X coordinate (along pitch axis)
# gains_thrust = [300, 0, 0] # Controls Y coordinate (along yaw axis)
# gains_pitch = [0, 0, 0] # controls Z coortinate (along roll axis)

# PID Gains list
pid_gains = [gains_roll, gains_thrust, gains_pitch]

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
    def arm_color(self, Color=None, Brightness=None):
        print_function_name()
    def reset_default_led(self):
        print_function_name()
    def get_battery_percentage():
        print_function_name()
        return(100)

class AutoPilot():
    def __init__(self, available=False, armed = False, enabled=False, timeout = 0.5, pid_gains = None, setpoint = None, output_limits = (-99,99)):
        self.available = False
        self.armed = False
        self.enabled = False
        self.timeout = timeout
        
        # Default PID gains
        if pid_gains is None:
            pid_gains = [np.array([0,0,0]) for i in range(3)]
        if setpoint is None:
            setpoint = np.zeros(len(pid_gains))

        # Enforce they have the same length
        if len(pid_gains) != len(setpoint):
            raise ValueError("AutoPilot.__init__(): pid_gains and setpoint must have same length")

        # PID controllers    
        self.pid = []
        for i, gains in enumerate(pid_gains):
            pid = PID(Kp = gains[0], Kd = gains[1], Ki = gains[2], sample_time = None)
            pid.output_limits = output_limits
            if setpoint is not None:
                pid.setpoint = setpoint[i]
            self.pid.append(pid)
        
        # Control outputs
        self.outputs = np.zeros(len(setpoint))

    def print_outputs(self):
        print(str(self.outputs))

    def update(self, state, dt = None):
        
        # Make sure state and setpoint have the same size
        if len(state) != len(self.pid):
            raise ValueError("AutoPilot.update(state): length of state vector must equal number of states")
        
        # Update the control outputs
        for i, s in enumerate(state): 
            self.outputs[i] = self.pid[i](state[i], dt = dt)
        return self.outputs

    def set_setpoint(self, setpoint):
        if len(setpoint) != len(self.pid):
            raise ValueError("AutoPilot.set_setpoint(setpoint): length of setpoint vector must equal number of states")

        # Update the setpoints    
        for i, s in enumerate(setpoint):
            self.pid[i].setpoint = s
            
    def off(self):
        for pid in self.pid:
            pid.auto_mode = False
    
    def on(self):
        for i, pid in enumerate(self.pid):
            pid.set_auto_mode(True, last_output=self.outputs[i])

    def reset(self):
        for pid in self.pid:
            pid.reset()
    
    def set_pid_gains(self, pid_gains):
        
        if len(pid_gains) != len(self.pid):
            raise ValueError("AutoPilot.set_pid_gains(pid_gains): length of pid_gains list must equal number of states")
        for i, pid in enumerate(self.pid):
            pid.Kp = pid_gains[i][0]
            pid.Kd = pid_gains[i][1]
            pid.Ki = pid_gains[i][2]

class AutoDrone(CoDrone.CoDrone):
    def __init__(self):
        super().__init__(self)
        self.autopilot = AutoPilot(available=False, armed = False, enabled=False)
        self.time = time.time()
        self.led_update_min_time = 0.2 # Default to only updating LEDs every 0.2 seconds
    
    def update_leds(self):
        now = time.time()
        dt = now - self.time
        if dt > self.led_update_min_time:
            if self.autopilot.enabled is True:
                # self.arm_color(CoDrone.Color.Blue, 100) # If the autopilot is enabled, arms steady blue
                self.arm_pattern(CoDrone.Color.Blue, CoDrone.Mode.DOUBLE_BLINK, 155)
                self.eye_color(CoDrone.Color.Blue, 100)
            elif self.autopilot.available is True:
                self.arm_pattern(CoDrone.Color.Blue, CoDrone.Mode.BLINK, 25)
                self.eye_pattern(CoDrone.Color.Blue, CoDrone.Mode.BLINK, 25)
            else:
                self.arm_pattern(CoDrone.Color.White, CoDrone.Mode.DOUBLE_BLINK, 155) # If autopilot isn't available
                self.eye_color(CoDrone.Color.White, 100)
            self.time = now

    def set_controls(self, state, dt=None):
        pid_outputs = self.autopilot.update(state, dt = dt)
        roll     =      pid_outputs[0] # roll controls x direction. -1 because x is flipped
        pitch    =      pid_outputs[2] # pitch controls movement in z direction
        throttle =      pid_outputs[1] # throttle controls altitude (y_xyz direction)

        if len(pid_outputs) > 3:
            yaw = pid_outputs[3]
        else:
            yaw = 0
        if self.is_flying():
            print("Autopilot [r,p,y,t] = %d,%d,%d,%d" % (roll, pitch, yaw, throttle))
            self.move(roll, pitch, yaw, throttle)
        controls_rpyt = [roll, pitch, yaw, throttle]
        return controls_rpyt

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

# Set up the connection for accepting incoming data
socket = 5555
sub = streams.Subscriber(socket=socket)
sub.connect()

# Initialize some toggle states
lb_down = False
lt_down = False
a_button_down = False
y_button_down = False
autopilot_led_on = False

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
    drone = AutoDrone()
    drone.autopilot.set_pid_gains(pid_gains = pid_gains)
    drone.autopilot.set_setpoint(setpoint = [0,0,0])

# Initialize pygame
pygame.init()
pygame.joystick.init()
print("Detected " + str(pygame.joystick.get_count()) + " joystick")
joystick = pygame.joystick.Joystick(0)

# Pressing "start" and "select" together pairs the drone
print("Hello, CoDrone")

# Get the stick bias
biasRoll,biasPitch,biasYaw,biasThrottle = get_controls(joystick, stick_sensitivity=1, scale_throttle = True)
biasRoll = joystick.get_axis(2)
biasPitch = joystick.get_axis(3)
biasYaw = joystick.get_axis(0)
biasThrottle = joystick.get_axis(1)
stickBias = controlInput(roll=biasRoll, pitch=biasPitch, yaw=biasYaw,throttle=biasThrottle)
print("Bias: " + str(stickBias))
print("Press START to pair.")

try:
    # When this flag is true, the program exits
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
    print("\tEmergency stop: START")
    print("\tToggle Autopilot: SELECT")
    print("\tQuit: START + SELECT")
    
    # Initialize stick sensitivity so user can't screw themselves up too badly with initial settings
    stick_sensitivity = np.clip(stick_sensitivity, 10, 100)

    # Timing stuff
    then = time.time()

    # Time since last autopilot frame
    msg_received_time_prev = 0

    # Set previous time step for autopilot state update
    autopilot_timestamp_prev = None

    # while drone.isConnected(): # Replace with while drone.isConnected()
    while drone.isConnected() and done is False:

        # Before we access the controls, grab any incoming state data off the zmq queue
        # msg = sub.receive_frame(blocking=False, return_dict=True)
        msg = sub.receive_last_frame()
        if msg is not None:
            msg_received_time_prev = time.time()
            if msg[CKeys.KF_ERROR_COV] < kf_err_cov_tracking_threshold:
                drone.autopilot.available = True
            else:
                drone.autopilot.available = False
                drone.autopilot.armed = False
                drone.autopilot.enabled = False
        else:
            time_since_last_autopilot_frame = time.time() - msg_received_time_prev
            if time_since_last_autopilot_frame > drone.autopilot.timeout: # If we don't get an autopilot signal within half a second, kill autopilot
                drone.autopilot.available = False
                drone.autopilot.armed = False
                drone.autopilot.enabled = False

        # Now read the control inputs
        for event in pygame.event.get(): # User did something.

            # Kill the autopilot if we get any control input from the gamepad
            if drone.autopilot.enabled is True and not selectPressed(joystick) and not event.type == pygame.JOYBUTTONUP:
                drone.autopilot.armed = False
                drone.autopilot.enabled = False
                print("Killing autopilot")
            
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
                        done = True

                # Select: Enable autopilot
                if selectPressed(joystick):
                    # Toggle arming the autopilot
                    drone.autopilot.armed = not(drone.autopilot.armed)
                
                # Left button down: Flash LEDs to indicate trimming
                if LB_PRESSED(joystick):
                    # drone.arm_pattern(CoDrone.Color.Blue, CoDrone.Mode.BLINK, 25)

                    # TODO: Update joystick trim info so that we can fine-tune trimming
                    lb_down = not lb_down
                
                # Left trigger down: reset trim
                if LT_PRESSED(joystick):
                    if lt_down is False:
                        trim.zero()
                        lt_down = not lt_down # Toggle the LT
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
                # TODO: Change this to set the trim when the button is pressed, not when it's released.
                if lb_down is True:
                    
                    # Get trim values
                    roll, pitch, yaw, throttle = get_controls(joystick, stick_sensitivity, bias=stickBias)
                    
                    # Set trim
                    trim.update(roll=roll,pitch=pitch,yaw=yaw,throttle=throttle)
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
                    print("Joystick [r,p,y,t] = %0.1f, %0.1f, %0.1f, %0.1f" % (roll, pitch, yaw, throttle))

        # Determine the autopilot state
        if drone.autopilot.available is True and drone.autopilot.armed is True:
            drone.autopilot.enabled = True
        else:
            drone.autopilot.enabled = False
            autopilot_timestamp_prev = None
            drone.autopilot.reset() #TODO: Should we just set the autopilot to "off" here?

        # If we got a message enabled, figure out the distance to the target location
        if msg is not None:

            # Read the timestamp and calculate the time interval
            timestamp_now = msg[CKeys.TIME_STAMP]
            lag = time.time() - timestamp_now
            print("Lag = %0.2e sec" % lag)
            if autopilot_timestamp_prev is None:
                autopilot_dt = None
            else:
                autopilot_dt = timestamp_now - autopilot_timestamp_prev
            
            # Set the previous time stamp
            autopilot_timestamp_prev = timestamp_now

            # Read the rotation and translation vectors 
            rvec = msg[CKeys.RVEC]
            tvec = msg[CKeys.TVEC]

            # Create homogeneous matrix to transform camera coordinates to drone coordinates
            pose_mat = np.eye(4)
            pose_mat[0:3, 0:3], _ = cv2.Rodrigues(rvec)
            pose_mat[0:3, 3] = np.transpose(tvec)

            # Transform xyz coordinates of target from camera coordinates to drone coordinates
            # Factors of -1 are because the vector product below gives the coordinates
            # of the target position in the frame of the drone, while the PID controller
            # needs the position of the drone in the frame of the set point.
            #
            # dx is not multipled by -1 because the x axis of the cube is in the wrong direction
            xyz_drone_target_frame_homogeneous = np.linalg.inv(pose_mat) @ xyz_target
            dx =  1 * xyz_drone_target_frame_homogeneous[0] / xyz_drone_target_frame_homogeneous[-1]
            dy = -1 * xyz_drone_target_frame_homogeneous[1] / xyz_drone_target_frame_homogeneous[-1]
            dz = -1 * xyz_drone_target_frame_homogeneous[2] / xyz_drone_target_frame_homogeneous[-1]

            # Print drone coordinates relative to set point
            # if autopilot_dt is not None:
                # print("[x,y,z] = [%0.0f, %0.0f, %0.0f] \\\ FPS = %0.1f \\\ lag = %0.2e sec" % (1000 * dx, 1000 * dy, 1000 * dz, 1 / autopilot_dt, time.time() - timestamp_now))
                # print("FPS: %0.1f, lag = %0.2e sec" % (1/autopilot_dt, time.time() - timestamp_now))

            # If the autopilot is enabled, set its controls
            if drone.autopilot.enabled == True:
                # Command the drone to move there
                ctrls_rpyt = drone.set_controls(state = np.array([dx, dy, dz]), dt = autopilot_dt)

                # print("<main> Autopilot [r,p,y,t] = %d, %d, %d, %d" % (ctrls_rpyt[0], ctrls_rpyt[1], ctrls_rpyt[2], ctrls_rpyt[3]))
                # drone.autopilot.print_outputs()
            
            # Print the coordinates in mm
            # print("Frame %d: [x, y, z] = %0.0f, %0.0f, %0.0f" % (msg[CKeys.FRAME_NUMBER], 1000 * dx, 1000 * dy, 1000 * dz))

        # End the timer
        now = time.time()
        fps = 1 / (now - then)
        then = now
        
        # Update the LEDs
        drone.update_leds()

except Exception as e:
    print("Exception raised: " + str(e))

finally:
    if drone.is_flying():
        print("Emergency stopping...")
        drone.emergency_stop()
        
    # Disconnect from the drone
    if drone.isConnected():
        batteryLevel = drone.get_battery_percentage()
        for x in range(10): # Spam this because it seems to not work every time
            drone.all_colors(CoDrone.Color.Red, 255)
        print("Battery: %d%%" % batteryLevel)
        print("Unpairing CoDrone...")
        for x in range(10):
            drone.disconnect()
        print("Unpaired CoDrone.")
    else:
        print("Warning: CoDrone connection dropped!!!")
        print("Done flag was " + str(done))
    print("Goodbye.")


