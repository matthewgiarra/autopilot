from simple_pid import PID
import numpy as np
import matplotlib.pyplot as plt
import time
from pdb import set_trace

def ThrustFromThrottle(throttle = 0):
    # Throttle should be -100 : 100
    thrust = 7.848E-4 * throttle + 0.363 # Newtons
    return thrust

# Formatting
xchar = "y(m)"
xdot = u"\u1E8F(m/s)"
xdotdot = u"\u00ff(m/s\u00B2)" 

# Gains p, d, i gains
# # Drive carefully 
# kp = 10
# kd = 50
# ki = 0

# MUST GO FASTER
kp = 200
kd = 180
ki = 25


# Mass is 39 grams
mass = 0.039 # Kg

# Target (set point) position
x_sp = 1 # meters

# Initial time
t = 0 # sec

# Time step (sec)
dt_mean = 0.05 # sec

# Time step standard deviation
s_dt = 2E-3

# Run time
# This particular pid library deals with
# time steps internally, and has to run in real time
run_time = 5 # sec


# PID object
pid = PID(Kp = kp, Ki = ki, Kd = kd)
pid.setpoint = x_sp
pid.output_limits = (-100, 100)

# Initial state
# x = 0 m, v = 0 m/s, a = 0 m/s^2
state = np.array([[0, 0, 0]]).T

# Initialize time variables
old_time = time.time() # sec
t_start = time.time() # sec
count = 0 # iterations

# Number of iterations to run
steps = run_time / dt_mean 

# Pre-allocate arrays for state variables and time
position = np.zeros(int(steps))
velocity = np.zeros(int(steps))
acceleration = np.zeros(int(steps))
t_sec = np.zeros(int(steps))

while count < steps:
# Get the current value
        
    # Calculate control input (force)
    control_input = pid(state[0,0]) # Percent (-100 to 100)

    # Resultant force from control input
    # From linear fit to measured data
    thrust = ThrustFromThrottle(control_input) # Newtons

    # Sum of forces
    force = thrust - mass * 9.81 # Newtons

    # Acceleration 
    accel = force / mass # m/s^2

    # Update acceleration in state matrix
    state[-1, 0] = accel # m/s^2

    # Print the state
    print("\t" + xchar + "\t" + xdot + "\t" + xdotdot)
    print("State: [%0.3f\t%0.3f\t%0.3f]" % tuple(state))
    print("Error: %0.3f" % (x_sp - state[0,0]) )
    print("Control input: %0.3f" % control_input)
    print("Net force: %0.2e N" % force)
    print("\n")

    # Time step
    dt = np.abs(dt_mean + s_dt * np.random.randn())
    F = np.array([[1, dt, 1/2 * dt * dt], [0, 1, dt], [0, 0, 1]])

    # Update state
    state = np.matmul(F, state)

    # Position
    position[count] = state[0,0] # m
    velocity[count] = state[1,0] # m/s
    acceleration[count] = state[2,0] # m/s^2

    t_sec[count] = time.time() - t_start # sec
    count += 1
    time.sleep(dt)

idx0 = 1 # Start index of the plot

fontSize = "xx-large"

# Make the figure
fig = plt.figure()
plt.plot(t_sec[idx0:], position[idx0:], '-k', t_sec[idx0:], velocity[idx0:], '-r', t_sec[idx0:], acceleration[idx0:], '-b', linewidth=3)
plt.plot(t_sec, x_sp * np.ones(t_sec.shape), '--k')
plt.legend([xchar, xdot, xdotdot], fontsize=fontSize)
plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.ylabel("State variable", fontsize=fontSize)
plt.xlabel("Time (sec)", fontsize=fontSize)

# Figure title
title_gains_str = r'$K_p=%0.1f, K_d = %0.1f, K_i=%0.1f$' % (kp, kd, ki)
fig.suptitle("Simulated system response with PID controller on vertical thrust\n1-D particle model with thrust and weight\n" + title_gains_str)

plt.show()



