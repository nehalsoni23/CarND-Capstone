
import math
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, car_params):
        # TODO: Implement
        self.car_params = car_params
        self.dbw_enabled = False
        self.prev_time = None
        self.brake_db = car_params.brake_db

        self.steer_filter = LowPassFilter(0.9, 0.8)
        self.yaw_controller = YawController(car_params.wheel_base,
                                            car_params.steer_ratio, car_params.min_speed,
                                            car_params.max_lat_accel, car_params.max_steer_angle)

        self.pid = PID(kp=0.7, ki=0.003, kd=0.1, mn=car_params.decel_limit, mx=car_params.accel_limit)

    def control(self, time, ref_lin_vel, ref_ang_vel, lin_vel, ang_vel, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs

        # TODO_NEHAL - Check if it works without self.dbw_enabled
        if not self.dbw_enabled and dbw_enabled:
            self.pid.reset()

        self.dbw_enabled = dbw_enabled

        if not dbw_enabled or self.prev_time is None:
            self.prev_time = time
            return 0.0, 0.0, 0.0

        delta_time = time - self.prev_time
        self.prev_time = time

        result = self.pid.step(ref_lin_vel - lin_vel, delta_time)

        if result > 0:
            throttle = result
            brake = 0.0
        elif math.fabs(result) > self.brake_db:
            throttle = 0.0
            brake = (self.car_params.vehicle_mass + (self.car_params.fuel_capacity * GAS_DENSITY)) * -result * self.car_params.wheel_radius
        else:
            throttle = 0.0
            brake = 0.0

        if ref_lin_vel == 0 and lin_vel < 0.5:
            throttle = 0.0
            temp_brake_val = (self.car_params.vehicle_mass + (self.car_params.fuel_capacity * GAS_DENSITY)) * -1.0 * self.car_params.wheel_radius
            brake = max(brake, temp_brake_val)

        steer = self.yaw_controller.get_steering(ref_lin_vel, ref_ang_vel, lin_vel)
        steer = self.steer_filter.filt(steer)

        
        # Return throttle, brake, steer
        return throttle, brake, steer
