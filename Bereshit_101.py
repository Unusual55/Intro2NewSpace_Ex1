'''
This class represents the basic flight controller of the Bereshit spacecraft.
'''

import math
import random
from builtins import float

import Moon

GR = 1.61803398875

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

    def __init__(self, alt, vs, hs, dist, fuel, ang):
        self.vs = vs
        self.hs = hs
        self.alt = alt
        self.dist = dist
        self.fuel = fuel
        self.ang = ang
        self.weight = Bereshit_101.WEIGHT_EMP + self.fuel
        self.time = 0
        self.dt = 1
        self.acc = 0
        self.NN = 0.7 + random.uniform(-0.1, 0.1)
        self.speed_update_rate = 0.003 + random.uniform(-0.001, 0.001)
        self.output = dict()
        self.simulated = False
        self.fitness = 0
        self.maintained_alt = 2000 + random.randrange(-250, 250)
        self.more_braking_power = 25 + random.randrange(0, 5)
        self.less_braking_power = 20 + random.randrange(-2, 2)
        self.slow_enough = 5
        self.output["vs_start"] = self.vs
        self.output["hs_start"] = self.hs
        self.output["fuel_start"] = self.fuel
        self.output["NN_start"] = self.NN
        self.output["more_braking_power_start"] = self.more_braking_power
        self.output["less_braking_power_start"] = self.less_braking_power
        self.output["maintained_alt_start"] = self.maintained_alt
        self.output["speed_update_rate"] = self.speed_update_rate


    def set_not_random(self):
        self.speed_update_rate = 0.003
        self.maintained_alt = 2000
        self.more_braking_power = 25
        self.less_braking_power = 20
        self.NN = 0.7
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

    def __str__(self):
        return (
            f'time: {self.time}, self.vs: {self.vs}, self.hs: {self.hs}, self.dist: {self.dist}, self.alt: {self.alt}'
            f', ang: {self.ang}, self.weight: {self.weight}, acc: {self.acc}, self.fuel: {self.fuel}')

    def simulate(self):
        # print("Simulating Bereshit's Landing:")

        # ***** main simulation loop ******
        while self.alt > 0:
            # if self.time % 10 == 0 or self.alt < 100:
            #     print(self)
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
                    self.NN = 1
                    if self.vs < self.slow_enough:
                        self.NN = 0.7  # if it is slow enough - go easy on the brakes

            if self.alt < 5 and self.vs < 5:  # no need to stop
                self.NN = 0.4

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
        self.fitness = ((2.5 - GR) * self.fuel - (GR * self.vs + self.hs))

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
        self.more_braking_power = other.output["more_braking_power_start"]
        self.less_braking_power = other.output["less_braking_power_start"]
        self.NN = other.output["NN_start"]
        self.maintained_alt = other.output["maintained_alt_start"]
        self.speed_update_rate = other.output["speed_update_rate"]


    def __eq__(self, other):
        if type(other) is not Bereshit_101:
            return False
        if self.output["more_braking_power_start"] != other.output["more_braking_power_start"]:
            return False
        if self.output["less_braking_power_start"] != other.output["less_braking_power_start"]:
            return False
        if self.output["NN_start"] == other.output["NN_start"]:
            return False
        if self.output["maintained_alt_start"] == other.output["maintained_alt_start"]:
            return False
        if self.output["speed_update_rate"] == other.output["speed_update_rate"]:
            return False
        return True

    def SinglePointCrossOver(self, parent):
        childA = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        childB = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        param = random.randrange(0, 5)
        childA.copy_data(self)
        childB.copy_data(parent)
        if param == 0:
            childA.NN = parent.output["NN_start"]
            childB.NN = self.output["NN_start"]
        if param == 1:
            childA.NN = parent.output["maintained_alt_start"]
            childB.NN = self.output["maintained_alt_start"]
        if param == 2:
            childA.NN = parent.output["speed_update_rate"]
            childB.NN = self.output["speed_update_rate"]
        if param == 3:
            childA.NN = parent.output["less_braking_power_start"]
            childB.NN = self.output["less_braking_power_start"]
        if param == 4:
            childA.NN = parent.output["more_braking_power_start"]
            childB.NN = self.output["more_braking_power_start"]
        childA.update_output_before_start()
        childB.update_output_before_start()
        return childA, childB

    def update_output_before_start(self):
        self.output["vs_start"] = self.vs
        self.output["hs_start"] = self.hs
        self.output["fuel_start"] = self.fuel
        self.output["NN_start"] = self.NN
        self.output["more_braking_power_start"] = self.maintained_alt
        self.output["less_braking_power_start"] = self.less_braking_power
        self.output["maintained_alt_start"] = self.more_braking_power
        self.output["speed_update_rate"] = self.speed_update_rate

    def MultiPointCrossOver(self, parent):
        param1 = random.randrange(0, 5)
        param2 = random.randrange(0, 5)
        while param1 == param2:
            param2 = random.randrange(0, 5)
        childA = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        childB = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        childA.copy_data(self)
        childB.copy_data(parent)
        if param1 == 0 or param2 == 0:
            childA.NN = parent.output["NN_start"]
            childB.NN = self.output["NN_start"]
        if param1 == 1 or param2 == 1:
            childA.maintained_alt = parent.output["maintained_alt_start"]
            childB.maintained_alt = self.output["maintained_alt_start"]
        if param1 == 2 or param2 == 2:
            childA.speed_update_rate = parent.output["speed_update_rate"]
            childB.speed_update_rate = self.output["speed_update_rate"]
        if param1 == 3 or param2 == 3:
            childA.less_braking_power = parent.output["less_braking_power_start"]
            childB.less_braking_power = self.output["less_braking_power_start"]
        if param1 == 4 or param2 == 4:
            childA.more = parent.output["more_braking_power_start"]
            childB.more_braking_power = self.output["more_braking_power_start"]
        childA.update_output_before_start()
        childB.update_output_before_start()
        return childA, childB

    def Mutate(self):
        mutated_childA = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        mutated_childB = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        mutated_childA.copy_data(self)
        mutated_childB.copy_data(self)
        param = random.randrange(0, 5)
        rate = random.uniform(-0.005, 0.005)
        #prevent rate == 0
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
        mutated_childA.update_output_before_start()
        mutated_childB.update_output_before_start()
        return mutated_childA, mutated_childB


def remove_duplicates(population):
    pop_set = set()
    for unit in poplulation:
        pop_set.add(unit)
    population.clear()
    for unit in pop_set:
        population.append(unit)
    return population

# 14095, 955.5, 24.8, 2.0
if __name__ == '__main__':

    # Create the first unit

    normal = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
    normal.set_not_random()
    normal.simulate()


    poplulation = []
    poplulation.append(normal)



    # Generate the first population

    for i in range(10000):
        spaceship = Bereshit_101(alt=13748, vs=24.8, hs=932, dist=181 * 1000, fuel=121, ang=58.3)
        spaceship.simulate()
        results = spaceship.get_results()
        if results["fitness"] > 0 and results["fuel"] > 0 and results["hs"] == 0 and results["vs"] < 10:
            poplulation.append(spaceship)


    # Generate 1000 Generations

    for gen in range(5000):
        for unit in poplulation:
            unit.get_results()
        poplulation = sorted(poplulation, key=lambda x: x.get_results()["fitness"], reverse=True)
        parentA = poplulation[0]
        results = parentA.get_results()
        print(
            f'Generation {gen} poplulation size: {len(poplulation)} best results: fitness:  {results["fitness"]:.12f}, vs: {results["vs"]}, hs: {results["hs"]}, fuel: {results["fuel"]}')
        parent_selection = random.randrange(0, 10)
        parentB = None
        if 0 <= parent_selection <= 7:
            parentB = poplulation[1]
        else:
            selection = random.randrange(1, len(poplulation))
            parentB = poplulation[selection]
        crossover_selection = random.randrange(0, 99)
        if 0 <= crossover_selection <= 80:
            childA, childB = parentA.SinglePointCrossOver(parentB)
        else:
            childA, childB = parentA.MultiPointCrossOver(parentB)
        poplulation.append(childA)
        poplulation.append(childB)
        mutation_selection = random.randrange(0, 100)
        if mutation_selection >= 70:
            mutated_child_A1, mutated_child_A2 = childA.Mutate()
            mutated_child_B1, mutated_child_B2 = childB.Mutate()
            poplulation.append(mutated_child_A1)
            poplulation.append(mutated_child_A2)
            poplulation.append(mutated_child_B1)
            poplulation.append(mutated_child_B2)

        # remove duplicates
        poplulation = remove_duplicates(poplulation)
