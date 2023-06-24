# GeneticAlgorithm
This project is experimental on Genetic Algorithms and how to improve their efficiency.


Since they can be used to solve complex problems with a space of possible solutions, Genetic algorithms are widely used in different fields such as speech recognition, computer networks, image processing, computer games and many more.

The traditional GA begins with initializing a population (P) with random individuals consisting of random genes. Next, depending on the problem to be solved, an evaluation equation is implemented to optimize results depending on the problem type. After evaluation, selection and mutation are implemented to result in a new population. This one cycle results in a generation which is repeated until the desired solution is reached. 


![image](https://github.com/Shadanjaradat/GeneticAlgorithm/assets/89996888/4d004816-2133-4815-9bdf-c13afc49e44d)



In this project, a genetic algorithm is implemented and then optimized examining the change of several parameters within this algorithm. The effect of parameter change such as mutation rates, mutation steps, population size, and selection methods such as tournament and roulette wheel selections was investigated throughout.


 # Findings

- Excessive generational runs can result in local optimum convergence, in which the fitness converges and gets stuck on a local optima rather than finding the global optimum.
- Diveristy within the genes in population is essential to avoid local optimum convergence.
- From the experimentation, it was concluded that small rates and steps (less than 0.01) and respectively larger ones (>2.8) are inefficient in GAs. Thus, during this project, the search space for these parameters was kept between 0.01 and 2.5 to reach for the best results.
- By comparing the three selection methods, tournament selection was the favoured one. This is mainly because the fittest individuals within a population were multi-sampled, and sometimes the whole population consists of these fitted individuals. Although this is beneficial as it leads to optimal solutions, the low diversity in the population sometimes led to early and non-optimal convergence.


# Future Works 

In later works, it would be interesting to try to design a system that automatically alternates between different selection methods while testing different values for rates and steps. This system would be more efficient as a wider space search would be implemented by changing multiple different parameters within the genetic algorithm.





