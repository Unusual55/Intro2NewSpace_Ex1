class PID_Controller:
    '''
    A PID implementation translated to python from  https://github.com/bryankate/simbeeotic/blob/master/simbeeotic-core/src/main/java/harvard/robobees/simbeeotic/util/PIDController.java

    Note that for time you should use nanoseconds:
    1. import time
    2. time.time_ns() <-- this is the input for the update method
    '''
    desired_val =   0.0

    p_gain      =   0.0
    i_gain      =   0.0
    d_gain      =   0.0

    prev_time   =   0.0
    prev_err    =   0.0
    integral    =   0.0

    def __init__ (self, desired_val, p_gain, i_gain, d_gain):
        self.desired_val = desired_val
        self.p_gain      = p_gain
        self.i_gain      = i_gain
        self.d_gain      = d_gain

    def setDesiredVal(self,desired_val):
        '''
        Sets the desired value/point and also resets the controller (previous values and integral)
        '''
        self.desired_val = desired_val
        self.reset()

    def reset(self):
        '''
        Resets the controller without setting a desired value/point
        '''
        self.prev_err  = 0.0
        self.prev_time = 0.0
        self.integral  = 0.0

    def update_desired_value(self, new_dv):
        if new_dv is not None:
            self.desired_val = new_dv

    def update(self, curr_time, curr_val) -> float:
        '''
        Calculates the current PID given the current time and value/point measurements
        '''

        if self.prev_time == 0.0: # first measurement
            self.prev_time = curr_time
            self.prev_err  = self.desired_val - curr_val
            return 0.0

        dt = curr_time - self.prev_time
        if dt == 0:
            return 0.0
        
        # Calculating the values for PID (without the gain factors)
        
        curr_error = self.desired_val - curr_val
        de = curr_error - self.prev_err # change in error
        derivative = de / dt 
        integral  += curr_error * dt 

        # setting this reading as the last one

        self.prev_time = curr_time
        self.prev_err  = curr_error

        # Calculating PID and returning a value

        P = self.p_gain * curr_error # "present"
        I = self.i_gain * integral   # "past"
        D = self.d_gain * derivative # "future"
        return P + I + D
    