from simple_pid import PID
import numpy as np
import matplotlib.pyplot as plt
import time
from pdb import set_trace

# Formatting
xdot = u"\u1E8B(m/s)"
xdotdot = u"\u1E8D(m/s\u00B2)" 

# Target (set point) position
x_sp = 0 # meters

# Initial time
t = 0 # sec

# Time step (sec)
dt_mean = 0.05 # sec

# Time step standard deviation
s_dt = 2E-3

# Run time
# This particular pid library deals with
# time steps internally, and has to run in real time
run_time = 20 # sec

# Gains p, d, i gains
# Drive carefully 
kp = 10
kd = 50
ki = 0

## MUST GO FASTER
# kp = 80
# kd = 120
# ki = 0

# Mass is 38 grams
mass = 0.038 # Kg

# Max force (thrust)
max_force = 0.05 # Newtons

# PID object
pid = PID(Kp = kp, Ki = ki, Kd = kd)

# Initial state
# x = 1 m, v = 0 m/s, a = 0 m/s^2
state = np.array([[1, 0, 0]]).T

# Initialize position array
errSum = 0 # meters
lastErr = 0 # meters

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
    force = control_input / 100 * max_force # Newtons

    # Acceleration 
    accel = force / mass # m/s^2

    # Update acceleration in state matrix
    state[-1, 0] = accel # m/s^2

    # Print the state
    print("\tx(m)\t" + xdot + "\t" + xdotdot)
    print("State: [%0.3f\t%0.3f\t%0.3f]" % tuple(state))
    print("Control input: %0.3f" % control_input)
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

# Make the figure
fig = plt.figure()
plt.plot(t_sec[idx0:], position[idx0:], '-k', t_sec[idx0:], velocity[idx0:], '-r', t_sec[idx0:], acceleration[idx0:], '-b')
plt.plot(t_sec, x_sp * np.ones(t_sec.shape), '--k')
plt.legend(["x(m)", xdot, xdotdot], fontsize="large")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.ylabel("State variable", fontsize="large")
plt.xlabel("Time (sec)", fontsize = "large")

# Figure title
title_gains_str = r'$K_p=%0.1f, K_d = %0.1f, K_i=%0.1f$' % (kp, kd, ki)
fig.suptitle("Simulated system response with PID controller on lateral thrust\n1-D particle model with thrust and weight\n" + title_gains_str)

plt.show()



