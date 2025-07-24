import matplotlib.pyplot as plt
import csv

generations = []
best_fitness = []
avg_fitness = []

with open('fitness_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        generations.append(int(row['generation']))
        best_fitness.append(float(row['best_fitness']))
        avg_fitness.append(float(row['avg_fitness']))

plt.figure(figsize=(10,6))
plt.plot(generations, best_fitness, label='Best Fitness')
plt.plot(generations, avg_fitness, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('NEAT Training Progress')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fitness_plot.png')
plt.show()
