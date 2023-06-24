# New Developed functions to be easily adjustable

import math
import random
import copy
import pandas as pd
import numpy as np

N = 8

P = 300
L = 3


# class that stores genes and fitness of individual in the population
class individual:
    def __init__(self):
        self.gene = []
        self.fitness = 0


population = []  # initialize an empty population


# function that creates a random population of P individuals with N genes
# This new function generates floats between 0 and 1
def populate_2(P, N, MIN, MAX):
    for i in range(P):
        new_ind = individual()

        for j in range(N):
            new_ind.gene.append(random.uniform(MIN, MAX))

        population.append(new_ind)

    return population


def evaluate(population):

    for i in range(P):

        population[i].fitness = fitness_ind(population[i], L)
        #for j in range(N):
            #population[i].gene.append(population[i][j])


def reset(population):
    for ind in range(P):
        ind = individual()
        ind.gene = None
        ind.fitness = None
    return 0


def fitness_ind(ind, L):
    f = 0

    if L == 1:
        for x in range(N):
            f = f + ind.gene[x]

    elif L == 2:
        f = 10 * N
        for i in range(N):
            f = f + ((ind.gene[i]) ** 2 - 10 * math.cos(2 * math.pi * ind.gene[i]))

    elif L == 3:
        part1 = 0.0
        part2 = 0.0
        mylist = ind.gene
        n = len(mylist)
        for i in range(0, n - 1):
            part1 += 100 * ((mylist[i + 1] - mylist[i] * mylist[i]) * (mylist[i + 1] - mylist[i] * mylist[i]))
            part2 += (1 - mylist[i]) * (1 - mylist[i])
        f = part1 + part2

    elif L == 4:

        sum1 = 0
        sum2 = 0

        for i in range(N):
            sum1 = sum1 + (ind.gene[i]) * (ind.gene[i])
            sum2 = sum2 + math.cos(2 * math.pi * ind.gene[i])

        sum1 = -0.2 * math.sqrt((1 / N) * sum1)
        sum2 = (1 / N) * sum2

        f = -20 * math.exp(sum1) - math.exp(sum2)
    elif L == 5:
        for x in range(N - 1):
            f = f + 100 * ((ind.gene[x + 1] - ind.gene[x] ** 2) ** 2) + (1 - ind.gene[x] ** 2)
    ind.fitness = f

    return f


def calc_pop_fit(input_pop, L):
    fitness = 0

    if L == 1:
        for ind in range(P):
            fitness = fitness + fitness_ind(input_pop[ind], 1)
            input_pop[ind].fitness = fitness_ind(input_pop[ind], L)

    elif L == 2:
        for ind in range(P):
            fitness = fitness + fitness_ind(input_pop[ind], 2)

    elif L == 3:
        for ind in range(P):
            fitness = fitness + fitness_ind(input_pop[ind], 3)
            input_pop[ind].fitness = fitness_ind(input_pop[ind], L)

    elif L == 4:
        for ind in range(P):
            fitness = fitness + fitness_ind(input_pop[ind], 4)

    return fitness


# find the ind with the best fitness
def find_best(input_pop, L):
    best = individual()
    best = input_pop[0]

    for ind in range(P):
        if fitness_ind(input_pop[ind], L) > fitness_ind(best, L):
            best = input_pop[ind]

    return best


def crossover(input_pop):
    tempoff1 = individual()
    tempoff2 = individual()
    temp = individual

    for b in range(0, P, 2):  # two steps so we can loop in pairs
        tempoff1 = input_pop[b]
        tempoff2 = input_pop[b + 1]
        temp = input_pop[b]
        crosspoint = random.randint(1, N)

        for c in range(crosspoint, N):
            tempoff1.gene[c] = tempoff2.gene[c]
            tempoff2.gene[c] = temp.gene[c]
        input_pop[b] = (tempoff1)
        input_pop[b + 1] = (tempoff2)

    return input_pop


# find the idv with the worst fitness
def find_worst(input_pop, L):
    worst = individual()
    worst = copy.deepcopy(input_pop[0])

    for ind in range(P):
        if fitness_ind(input_pop[ind], L) < fitness_ind(worst, L):
            worst = copy.deepcopy(input_pop[ind])

    return worst


# calculate the mean fitness of the population
def calc_mean(input_pop, L):
    mean = calc_pop_fit(input_pop, L) / P
    return mean


def roulette_wheel_selection(pop, L):
    offspring = []

    # roulette wheel selection for maximizing functions
    for i in range(P):

        # select random number

        random_selection = random.uniform(0, 1 / calc_pop_fit(pop, L))

        running_total = 0
        j = 0

        while running_total <= random_selection:
            running_total += pop[j].fitness
            j = j + 1

        offspring.append(pop[j - 1])

    return offspring


def tournament_selection(input_pop, L):
    offspring = []

    for i in range(0, P):

        parent1 = random.randint(0, P - 1)  # pick an individual randomly from population
        off1 = copy.deepcopy(input_pop[parent1])

        parent2 = random.randint(0, P - 1)
        off2 = copy.deepcopy(input_pop[parent2])

        if off1.fitness < off2.fitness:
            offspring.append(off1)

        else:
            offspring.append(off2)

    return offspring


# like tournament selection, but 10 offsprings rather than 2
def kay_selection(input_pop, L):
    offspring = []
    offspring.clear()
    off = []
    off.clear()
    for i in range(0, P):

        parent1 = random.randint(0, P - 1)  # pick an individual randomly from population
        off.append(input_pop[parent1])

        parent2 = random.randint(0, P - 1)
        off.append(input_pop[parent2])

        parent3 = random.randint(0, P - 1)  # pick an individual randomly from population
        off.append(input_pop[parent3])

        parent4 = random.randint(0, P - 1)
        off.append(input_pop[parent4])

        parent5 = random.randint(0, P - 1)  # pick an individual randomly from population
        off.append(input_pop[parent5])

        parent6 = random.randint(0, P - 1)
        off.append(input_pop[parent6])

        parent7 = random.randint(0, P - 1)  # pick an individual randomly from population
        off.append(input_pop[parent7])

        parent8 = random.randint(0, P - 1)
        off.append(input_pop[parent8])

        parent9 = random.randint(0, P - 1)  # pick an individual randomly from population
        off.append(input_pop[parent9])

        parent10 = random.randint(0, P - 1)
        off.append(input_pop[parent10])

        for j in range(10):

            min = off[0]

            if fitness_ind(off[j], L) <= fitness_ind(min, L):
                min = off[j]

        offspring.append(min)

    return offspring


# New function that mutates the float numbers genes
# bounds the genes between max and min
def mutate_float(offspring, MAX, MIN, MUTRATE, MUTSTEP):
    new_population = []

    for i in range(0, P):

        new_ind = individual()

        new_ind.gene = []

        for j in range(0, N):

            gene = offspring[i].gene[j]
            mut_prob = random.random()

            if mut_prob < MUTRATE:
                alter = random.gauss(0, MUTSTEP)  # alter between 0 and mutstep

                gene = gene - 2 * alter

                if gene > MAX:
                    gene = MAX

                if gene < MIN:
                    gene = MIN

            new_ind.gene.append(gene)

        new_population.append(new_ind)

    return new_population


# FUNC = 0 if we are minimizing, FUN = 1 if we are maximizing
def best_accuracy(list, FUNC):
    x = 0
    if FUNC == 0:
        for i in range(P):
            if list[i + 1] <= list[i]:
                x = x + 1
    if FUNC == 1:
        for i in range(P):
            if list[i + 1] >= list[i]:
                x = x + 1

    error = 100 * ((P - x) / P)
    print("Error Rate: %s" % error, "%")
    return error


def best_parameters(bestinpop, MUTSTEPS, MUTRATES):
    min = bestinpop[0]
    beststep = MUTSTEPS[0]
    bestrate = MUTRATES[0]

    for i in range(500):
        if bestinpop[i] <= min:
            min = bestinpop[i]
            beststep = MUTSTEPS[i]
            bestrate = MUTRATES[i]

    print("Best fitness: ", min)
    print("Best mutrate: ", bestrate)
    print("Best mutstep: ", beststep)


def data_excel(MUTRATES, MUTSTEPS, bestinpop, R):
    # to excel
    df = pd.DataFrame({'MUTRATE': MUTRATES, 'MUTSTEP': MUTSTEPS,
                       'Best': bestinpop})

    if R == 0:
        writer = pd.ExcelWriter('3run1.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()
    elif R == 1:
        writer = pd.ExcelWriter('3run2.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()
    elif R == 2:
        writer = pd.ExcelWriter('3run3.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()
    elif R == 3:
        writer = pd.ExcelWriter('3run4.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()

    elif R == 4:
        writer = pd.ExcelWriter('3run5.xlsx', engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()
    elif R == 5:
        writer = pd.ExcelWriter('3run6.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.

        df.to_excel(writer, sheet_name='Sheet2', index=False)

        writer.save()
