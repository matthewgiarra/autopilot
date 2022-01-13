import CoDrone
import cv2
import numpy as np
import time
import csv

drone_id = 8898
blank = np.zeros(shape=[64,64,3], dtype=np.uint8)

throttle = 10
roll = 0
pitch = 0
yaw = 0

# Instantiate drones
drone = CoDrone.CoDrone()

# Establish bluetooth connection
if not drone.isConnected():
    drone.pair(str(drone_id))
try:
    # drone.takeoff()
    start_time = time.time()
    t_old = time.time()
    
    # Go full throttle
    print("Going full throttle....")
    drone.set_throttle(throttle)
    drone.move(roll, pitch, yaw, throttle)
    drone.move()

    with open("battery_data.txt", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(["time (sec)", "throttle", "battery volts", "battery percent"])
        while True:
            t_sec = time.time() - start_time
            dt = t_sec - t_old
            t_old = t_sec
            battery_voltage = drone.get_battery_voltage() / 100
            battery_percentage = drone.get_battery_percentage()
            current_throttle = drone.get_throttle()
            print("%0.1f sec: Battery %0.2f V (%0d%%), throttle %d" % (round(t_sec, 1), battery_voltage, battery_percentage, current_throttle))

            csvwriter.writerow([round(t_sec, 1), current_throttle, battery_voltage, battery_percentage])
            time.sleep(1)

except Exception as e:
    print("Exception raised: " + str(e))
finally:
    if drone.is_flying():
        print("Emergency stopping...")
        drone.emergency_stop()
        
    # Spam this to make sure it disconnects!
    print("Disconnecting...")
    drone.disconnect()
    print("Disconnected CoDrone.")
print("Done.")



