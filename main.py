import copy
import matplotlib.pyplot as plt
import pandas as pd

from Functions import P
from Functions import N

from Functions import populate_2
from Functions import fitness_ind
from Functions import calc_pop_fit
from Functions import find_best
from Functions import find_worst
from Functions import calc_mean
from Functions import roulette_wheel_selection
from Functions import tournament_selection
from Functions import mutate_float
from Functions import best_accuracy
from Functions import best_parameters
from Functions import data_excel
import numpy as np
import random
from Functions import crossover
from mpl_toolkits import mplot3d
from Functions import kay_selection
mean_list = []
best_list = []

Generations = 50


MAX = 100
MIN = -100
L = 3

# minimization values


RUN = 5

MUTSTEPS = []
bestinpop = []
worstinpop = []
MUTRATES = []
population = []


LowerStepRange = 0.01
HigherStepRange = 2.5

LowerRateRange = 0.01
HigherRateRange = 0.05




# run the whole code RUN times
for R in range(0, RUN):
    bestinpop.clear()
    MUTSTEPS.clear()
    MUTRATES.clear()


    for y in range(500):

        MUTRATE = random.uniform(LowerRateRange, HigherRateRange)
        MUTSTEP = random.uniform(LowerStepRange, HigherStepRange)
        population.clear()
        population = populate_2(P, N, MIN, MAX)
        calc_pop_fit(population, L)

        for i in range(Generations):

            offspring = []
            offspring.clear()
            fitness_list = []
            fitness_list.clear()

            # 1. select parents && recombine pairs
            offspring = tournament_selection(population, L)

            new_population = crossover(offspring)
            # 2. mutate the offspring
            new_population = mutate_float(new_population, MAX, MIN, MUTRATE, MUTSTEP)

            # find the worst individual of the original population
            worst = find_worst(population, L)

            # 3. if new pop is less fit than the old one, replace it using this
            main = copy.deepcopy(population)

            if calc_pop_fit(new_population, L) < calc_pop_fit(population, L):
                population = copy.deepcopy(new_population)

            mean = calc_mean(population, L)
            mean_list.append(mean)

            # find the best individual of the new population
            best = find_best(population, L)

            # remove best from population, and append worst instead
            population.append(worst)
            population.remove(best)

            best = fitness_ind(find_best(population, L), L)
            # worst = fitness_ind(find_worst(population, L), L)
            best_list.append(best)
            calc_pop_fit(population, L)

        # find best of the final population (generation)
        bestind = fitness_ind(find_best(population, L), L)
        bestind2 = fitness_ind(find_worst(population, L), L)
        MUTSTEPS.append(MUTSTEP)
        MUTRATES.append(MUTRATE)
        bestinpop.append(bestind2)
        print(MUTRATE, MUTSTEP, bestind2)

    best_parameters(bestinpop, MUTSTEPS, MUTRATES)
    # data_excel(MUTRATES, MUTSTEPS, bestinpop, R)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# plot3(MUTRATES, MUTSTEPS, bestinpop)
my_cmap = plt.get_cmap('hsv')

sctt = ax.scatter(MUTRATES, MUTSTEPS, bestinpop, cmap=my_cmap, marker='.')
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.show()
