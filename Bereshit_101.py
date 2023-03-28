'''
This class represents the basic flight controller of the Bereshit spacecraft.
'''

import math,random,time
from builtins import float
import matplotlib.pyplot as plt
import numpy as np

import Moon
from PID import PID_Controller as pid

GR = 1.61803398875 # Golden Ration
INIT_DVS = 24


class Bereshit_101:
    # All this variable type is float.
    WEIGHT_EMP = 165  # kg
    WEIGHT_FULE = 420  # kg
    WEIGHT_FULL = WEIGHT_EMP + WEIGHT_FULE  # kg
    # https://davidson.weizmann.ac.il/online/askexpert/%D7%90%D7%99%D7%9A-%D7%9E%D7%98%D7%99%D7%A1%D7%99%D7%9D-%D7%97%D7%9C%D7%9C%D7%99%D7%AA-%D7%9C%D7%99%D7%A8%D7%97
    MAIN_ENG_F = 430  # N
    SECOND_ENG_F = 25  # N
    MAIN_BURN = 0.15  # liter per sec, 12 liter per m'
    SECOND_BURN = 0.009  # liter per sec 0.6 liter per m'
    ALL_BURN = MAIN_BURN + 8 * SECOND_BURN

    def __init__(self, alt, vs, hs, dist, fuel, ang, p_gain, i_gain, d_gain, desired_val):
        self.vs = vs
        self.hs = hs
        self.alt = alt
        self.dist = dist
        self.fuel = fuel
        self.ang = ang
        self.input_parameters = {"vs": vs, "hs": hs, "dist": dist, "alt": alt, "fuel": fuel, "ang": ang,
                                 "desired_val": desired_val,"p_gain": p_gain,"i_gain": i_gain, "d_gain": d_gain}
        self.weight = Bereshit_101.WEIGHT_EMP + self.fuel
        self.time = 0
        self.dt = 0.25 # was 1.0
        self.acc = 0
        self.NN = 0.7 + random.uniform(-0.1, 0.1)
        self.speed_update_rate = 0.003 + random.uniform(-0.001, 0.001)
        self.output = dict()
        self.random_parameters = dict()
        self.simulated = False
        self.fitness = 0
        self.maintained_alt = 2000 + random.randrange(-250, 250)
        self.more_braking_power = 25 + random.randrange(0, 5)
        self.less_braking_power = 20 + random.randrange(-2, 2)
        self.slow_enough = 5 + random.randrange(-2, 4)
        self.output["vs_start"] = self.vs
        self.output["hs_start"] = self.hs
        self.output["fuel_start"] = self.fuel
        self.random_parameters["NN"] = self.NN
        self.random_parameters["more_braking_power"] = self.more_braking_power
        self.random_parameters["less_braking_power"] = self.less_braking_power
        self.random_parameters["maintained_alt"] = self.maintained_alt
        self.random_parameters["speed_update_rate"] = self.speed_update_rate
        self.random_parameters["slow_enough"] = self.slow_enough
        self.init_pid(desired_val=desired_val, p_gain=p_gain, i_gain=i_gain, d_gain=d_gain)
        self.dvs = INIT_DVS
        self.landing_slope = -1
        self.alt_log  = []
        self.vs_log   = []
        self.time_log = []
        self.fuel_log = []

    def update_logs(self):
        self.alt_log.append(self.alt)
        self.vs_log.append(self.vs)
        self.time_log.append(self.time)
        self.fuel_log.append(self.fuel)

    def init_pid(self, p_gain, i_gain, d_gain, desired_val):
        self.p_gain = p_gain
        self.i_gain = i_gain 
        self.d_gain = d_gain
        self.pid = pid(desired_val=desired_val, p_gain=p_gain, i_gain=i_gain, d_gain=d_gain,)

    def set_not_random(self):
        self.speed_update_rate = 0.003
        self.maintained_alt = 2000
        self.more_braking_power = 25
        self.less_braking_power = 20
        self.NN = 0.7
        self.slow_enough = 5
        self.update_output_before_start()

    def accMax(self, weight: float) -> float:
        return self.acc_fun(weight, True, 8)

    def acc_fun(self, weight: float, main: bool, seconds: int) -> float:
        t = 0
        if main:
            t += Bereshit_101.MAIN_ENG_F
        t += seconds * Bereshit_101.SECOND_ENG_F
        ans = t / weight
        return ans

    def get_info(self):
        return (
            f'time: {self.time}, self.vs: {self.vs}, self.hs: {self.hs}, self.dist: {self.dist}, self.alt: {self.alt}'
            f', ang: {self.ang}, self.weight: {self.weight}, acc: {self.acc}, self.fuel: {self.fuel}')

    
    def naive_dvs(self):
        if self.alt > 700:
            return INIT_DVS
        if self.alt > 500:
            return 16
        if self.alt > 300:
            return 12
        if self.alt > 200:
            return 6
        if self.alt > 100:
            return 3
        return 1
    
    def linear_fun(self, m, x, b):
        '''
        y = m*x +b

        for this program:
        
        x axis - self.dvs,
        m      - self.landing_slope,
        b      - default 0
        '''
        y = m * x + b
        return y
    
    def naive_loop(self):
        if self.time % 10 == 0 or self.alt < 100:
             print(self.get_info())
        # over 2 km above the ground
        if self.alt > self.maintained_alt:  # maintain a vertical speed of [20-25] m/s
            if self.vs > self.more_braking_power:
                self.NN += self.speed_update_rate * self.dt
            # more power for braking
            if self.vs < self.less_braking_power:
                self.NN -= self.speed_update_rate * self.dt
            # less power for braking
        # lower than 2 km - horizontal speed should be close to zero
        else:
            if self.ang > 3:
                self.ang -= 3
            # rotate to vertical position.
            else:
                self.ang = 0
            self.NN = 0.5  # brake slowly, a proper PID controller here is needed!
            if self.hs < 2:
                self.hs = 0
            if self.alt < 125:  # very close to the ground!
                self.NN = 0.9
                if self.vs < self.slow_enough:
                    self.NN = 0.7  # if it is slow enough - go easy on the brakes
        if self.alt < 5:  # no need to stop
            self.NN = 0.4

    def pid_loop(self):
        self.update_logs()

        if self.alt > self.maintained_alt:
            '''
            if self.vs > self.more_braking_power:
                self.NN += self.speed_update_rate * self.dt # more power for braking
            if self.vs < self.less_braking_power:
                self.NN -= self.speed_update_rate * self.dt # less power for braking
            '''
        else:
            if self.landing_slope == -1:
                self.landing_slope = self.vs / self.alt

            if self.ang > 3:
                self.ang -= 3
            # rotate to vertical position.
            else:
                self.ang = 0

            try:
                new_dvs = self.linear_fun(x = self.alt, m = self.landing_slope, b = 1)
                self.pid.update_desired_value(new_dvs)
                pid_out = self.pid.update(curr_time=time.time(),curr_val=self.alt)
                self.NN += pid_out
            except:
                self.NN = 0.5

            if self.hs < 2:
                self.hs = 0
            if self.alt < 125:  # very close to the ground!
                self.NN = 0.9
                if self.vs < self.slow_enough:
                    self.NN = 0.7  # if it is slow enough - go easy on the brakes
        if self.alt < 5:  # no need to stop
            self.NN = 0.4

    def sigmoid(num):
        return 1 / (1 + math.pow(math.e,-1.0*num))

    def simulate(self):
        # print("Simulating Bereshit's Landing:")
        # ***** main simulation loop ******
        while self.alt > 0:
            if self.pid is None:
                self.naive_loop()
            else:
                self.pid_loop()

            # main computations
            ang_rad = math.radians(self.ang)
            h_acc = math.sin(ang_rad) * self.acc
            v_acc = math.cos(ang_rad) * self.acc
            vacc = Moon.getAcc(self.hs)
            self.time += self.dt
            dw = self.dt * Bereshit_101.ALL_BURN * self.NN

            if self.fuel > 0:
                self.fuel -= dw
                self.weight = Bereshit_101.WEIGHT_EMP + self.fuel
                self.acc = self.NN * self.accMax(self.weight)
            else:  # ran out of self.fuel
                self.acc = 0

            v_acc -= vacc

            if self.hs > 0:
                self.hs -= h_acc * self.dt

            self.dist -= self.hs * self.dt
            self.vs -= v_acc * self.dt
            self.alt -= self.dt * self.vs

        # Data for genetics algorithms

        self.output["vs_end"] = self.vs
        self.output["hs_end"] = self.hs
        self.output["fuel_end"] = self.fuel
        self.simulated = True
        self.fitness = self.my_fitness()
        # if self.vs > 2.5 and self.fitness > 0:
        #     self.fitness /= 2
        # print(self.get_info())

    def my_fitness(self):
        return ((2.5 - GR) * self.fuel - (GR * self.vs + self.hs))

    def __str__(self):
        return str(self.output)

    def __hash__(self):
        return hash(str(self))

    def get_results(self):
        if self.simulated is True:
            return {'fitness': self.fitness, 'vs': self.output['vs_end'], 'hs': self.output['hs_end'],
                    'fuel': self.output['fuel_end']}
        else:
            self.simulate()
            self.get_results()

    def copy_data(self, other):
        self.more_braking_power = other.random_parameters["more_braking_power"]
        self.less_braking_power = other.random_parameters["less_braking_power"]
        self.NN = other.random_parameters["NN"]
        self.maintained_alt = other.random_parameters["maintained_alt"]
        self.speed_update_rate = other.random_parameters["speed_update_rate"]
        self.slow_enough = other.random_parameters["slow_enough"]

    def __eq__(self, other):
        if type(other) is not Bereshit_101:
            return False
        if self.random_parameters["more_braking_power"] != other.random_parameters["more_braking_power"]:
            return False
        if self.random_parameters["less_braking_power"] != other.random_parameters["less_braking_power"]:
            return False
        if self.random_parameters["NN"] == other.random_parameters["NN"]:
            return False
        if self.random_parameters["maintained_alt"] == other.random_parameters["maintained_alt"]:
            return False
        if self.random_parameters["speed_update_rate"] == other.random_parameters["speed_update_rate"]:
            return False
        if self.random_parameters["slow_enough"] == other.random_parameters["slow_enough"]:
            return False
        return True

    def SinglePointCrossOver(self, parent):
        if self.input_parameters["p_gain"] is None:
            print("YES")
            exit
        childA = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                              hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                              fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        childB = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                              hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                              fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        param = random.randrange(0, 6)
        childA.copy_data(self)
        childB.copy_data(parent)
        if param == 0:
            childA.NN = parent.random_parameters["NN"]
            childB.NN = self.random_parameters["NN"]
        if param == 1:
            childA.maintained_alt = parent.random_parameters["maintained_alt"]
            childB.maintained_alt = self.random_parameters["maintained_alt"]
        if param == 2:
            childA.speed_update_rate = parent.random_parameters["speed_update_rate"]
            childB.speed_update_rate = self.random_parameters["speed_update_rate"]
        if param == 3:
            childA.less_braking_power = parent.random_parameters["less_braking_power"]
            childB.less_braking_power = self.random_parameters["less_braking_power"]
        if param == 4:
            childA.more = parent.random_parameters["more_braking_power"]
            childB.more_braking_power = self.random_parameters["more_braking_power"]
        if param == 5:
            childA.slow_enough = parent.random_parameters["slow_enough"]
            childB.slow_enough = self.random_parameters["slow_enough"]
        childA.update_output_before_start()
        childB.update_output_before_start()
        return childA, childB

    def update_output_before_start(self):
        self.output["vs_start"] = self.vs
        self.output["hs_start"] = self.hs
        self.output["fuel_start"] = self.fuel
        self.random_parameters["NN"] = self.NN
        self.random_parameters["more_braking_power"] = self.maintained_alt
        self.random_parameters["less_braking_power"] = self.less_braking_power
        self.random_parameters["maintained_alt"] = self.more_braking_power
        self.random_parameters["speed_update_rate"] = self.speed_update_rate
        self.random_parameters["slow_enough"] = self.slow_enough

    def MultiPointCrossOver(self, parent):
        param1 = random.randrange(0, 6)
        param2 = random.randrange(0, 6)
        while param1 == param2:
            param2 = random.randrange(0, 6)
        if self.input_parameters["p_gain"] is None:
            print("YES")
            exit
        childA = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                              hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                              fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        childB = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                              hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                              fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        childA.copy_data(self)
        childB.copy_data(parent)
        if param1 == 0 or param2 == 0:
            childA.NN = parent.random_parameters["NN"]
            childB.NN = self.random_parameters["NN"]
        if param1 == 1 or param2 == 1:
            childA.maintained_alt = parent.random_parameters["maintained_alt"]
            childB.maintained_alt = self.random_parameters["maintained_alt"]
        if param1 == 2 or param2 == 2:
            childA.speed_update_rate = parent.random_parameters["speed_update_rate"]
            childB.speed_update_rate = self.random_parameters["speed_update_rate"]
        if param1 == 3 or param2 == 3:
            childA.less_braking_power = parent.random_parameters["less_braking_power"]
            childB.less_braking_power = self.random_parameters["less_braking_power"]
        if param1 == 4 or param2 == 4:
            childA.more = parent.random_parameters["more_braking_power"]
            childB.more_braking_power = self.random_parameters["more_braking_power"]
        if param1 == 5 or param2 == 5:
            childA.slow_enough = parent.random_parameters["slow_enough"]
            childB.slow_enough = self.random_parameters["slow_enough"]
        childA.update_output_before_start()
        childB.update_output_before_start()
        return childA, childB

    def Mutate(self):
        mutated_childA = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                                      hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                                      fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        mutated_childB = Bereshit_101(alt=self.input_parameters["alt"], vs=self.input_parameters["vs"],
                                      hs=self.input_parameters["hs"], dist=self.input_parameters["dist"],
                                      fuel=self.input_parameters["fuel"], ang=self.input_parameters["ang"],
                              p_gain=self.input_parameters["p_gain"],i_gain=self.input_parameters["i_gain"],
                              d_gain=self.input_parameters["d_gain"],desired_val=self.input_parameters["desired_val"])
        mutated_childA.copy_data(self)
        mutated_childB.copy_data(self)
        params = [x for x in range(0, 6)]
        rate = 0.05
        # prevent rate == 0
        while len(params) > 2:
            param = random.choice(params)
            params.remove(param)
            if param == 0:
                mutated_childA.NN *= (1 + rate)
                mutated_childB.NN *= (1 - rate)
            if param == 1:
                mutated_childA.maintained_alt *= (1 + rate)
                mutated_childB.maintained_alt *= (1 - rate)
            if param == 2:
                mutated_childA.speed_update_rate *= (1 + rate)
                mutated_childB.speed_update_rate *= (1 - rate)
            if param == 3:
                mutated_childA.less_braking_power *= (1 + rate)
                mutated_childB.less_braking_power *= (1 - rate)
            if param == 4:
                mutated_childA.more_braking_power *= (1 + rate)
                mutated_childB.more_braking_power *= (1 - rate)
            if param == 5:
                mutated_childA.slow_enough *= (1 + rate)
                mutated_childB.slow_enough *= (1 - rate)
        mutated_childA.update_output_before_start()
        mutated_childB.update_output_before_start()
        return mutated_childA, mutated_childB


def remove_duplicates(population):
    pop_set = set()
    for unit in population:
        pop_set.add(unit)
    population.clear()
    for unit in pop_set:
        population.append(unit)
    return population


# 14095, 955.5, 24.8, 2.0
if __name__ == '__main__':

    # Create the first unit
    alt = 13748
    vs = 24.8
    hs = 932
    dist = 181 * 1000
    fuel = 121
    ang = 58.3
    p_gain = 0.04
    i_gain = 0.0003
    d_gain = 0.2
    population_bound = 500
    generation_bound = 1000
    normal = Bereshit_101(alt=alt, vs=vs, hs=hs, dist=dist, fuel=fuel, ang=ang,
                           p_gain=p_gain, i_gain=i_gain,d_gain=d_gain, desired_val= INIT_DVS)
    normal.set_not_random()
    normal.simulate()
    #
    population = []
    population.append(normal)
    population_best = []

    # plt.ion()



    # Generate the first population

    while len(population) < population_bound:
        spaceship = Bereshit_101(alt=alt, vs=vs, hs=hs, dist=dist, fuel=fuel, ang=ang,
                                  p_gain=p_gain, i_gain=i_gain,d_gain=d_gain, desired_val= INIT_DVS)
        spaceship.simulate()
        results = spaceship.get_results()
        if results["fitness"] > 0 and results["fuel"] > 0 and results["vs"] < 10 and results["hs"] == 0:
            population.append(spaceship)

    best_sample = normal
    X_pop = [0]
    Y_pop = [len(population)]
    X_fuel = []
    Y_fuel = []
    X_fit = []
    Y_fit = []

    # Generate 1000 Generations

    for gen in range(generation_bound):
        for unit in population:
            unit.get_results()
        population = sorted(population, key=lambda x: x.get_results()["fitness"], reverse=True)
        population_best = [x for x in population if x.get_results()["vs"] <= 2.5 and x.get_results()["hs"] == 0]
        population_best = sorted(population_best, key=lambda x: x.get_results()["fuel"], reverse=True)
        X_fuel.append(gen)
        Y_fuel.append(population_best[0].get_results()["fuel"])
        # parentA = population_best[0]
        parentA = population[0]
        results = parentA.get_results()
        X_fit.append(gen)
        Y_fit.append(parentA.get_results()["fitness"])

        X_pop.append(gen + 1)
        Y_pop.append(len(population))

        print(
            f'Generation {gen} population size: {len(population)} best results: fitness:  {results["fitness"]:.12f}, vs: {results["vs"]}, hs: {results["hs"]}, fuel: {results["fuel"]}')
        parent_selection = random.randrange(0, 10)
        parentB = None
        if 0 <= parent_selection <= 7:
            parentB = population[1]
        # elif 5 <= parent_selection <= 8:
        #     parentB = population_best[0]
        else:
            selection = random.randrange(1, len(population))
            parentB = population[selection]
        crossover_selection = random.randrange(0, 99)
        if 0 <= crossover_selection <= 50:
            childA, childB = parentA.SinglePointCrossOver(parentB)
        else:
            childA, childB = parentA.MultiPointCrossOver(parentB)
        population.append(childA)
        population.append(childB)
        mutation_selection = random.randrange(0, 100)
        if mutation_selection >= 69:
            mutated_child_A1, mutated_child_A2 = childA.Mutate()
            mutated_child_B1, mutated_child_B2 = childB.Mutate()
            population.append(mutated_child_A1)
            population.append(mutated_child_A2)
            population.append(mutated_child_B1)
            population.append(mutated_child_B2)

        # remove duplicates
        population = remove_duplicates(population)
        if gen == generation_bound-1:
            best_sample = parentA

    print(len(population))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    line_fitness = ax1.plot(best_sample.time_log, best_sample.vs_log, 'b-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Vertical Speed')

    alt_plot = ax2.plot(best_sample.time_log, best_sample.alt_log, 'r')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Altitude')

    fuel_plot = ax3.plot(best_sample.time_log, best_sample.fuel_log)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fuel')

    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    ax3.relim()
    ax3.autoscale_view()

    plt.show()
