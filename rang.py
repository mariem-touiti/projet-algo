import random
import math

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale d'une solution (route complète)"""
    distance_totale = 0
    for i in range(len(solution) - 1):
        distance_totale += matrice_distances[solution[i]][solution[i + 1]]
    distance_totale += matrice_distances[solution[-1]][solution[0]]
    return distance_totale


def creer_population_initiale(taille_population, nombre_villes):
    """Crée une population initiale de solutions aléatoires"""
    population = []
    for _ in range(taille_population):
        individu = list(range(nombre_villes))
        random.shuffle(individu)
        population.append(individu)
    return population


def calculer_fitness(individu, matrice_distances):
    """Calcule le fitness (inverse de la distance pour maximiser)"""
    distance = calculer_distance_totale(individu, matrice_distances)
    return 1 / distance if distance > 0 else 0


def selection_rang(population, fitnesses):
    """Sélection par rang: probabilité basée sur le classement"""
    indices_fitness = [(i, f) for i, f in enumerate(fitnesses)]
    
    indices_fitness.sort(key=lambda x: x[1])
    
    rangs = {}
    for rang, (idx, _) in enumerate(indices_fitness, start=1):
        rangs[idx] = rang
    
    somme_rangs = sum(rangs.values())
    
    probabilites_cumulatives = []
    cumul = 0
    for i in range(len(population)):
        cumul += rangs[i] / somme_rangs
        probabilites_cumulatives.append(cumul)
    
    r = random.random()
    for i, prob_cumul in enumerate(probabilites_cumulatives):
        if r <= prob_cumul:
            return population[i][:]
    
    return population[-1][:]


def croisement_pmx(parent1, parent2):
    """Croisement PMX (Partially Mapped Crossover)"""
    taille = len(parent1)
    point1, point2 = sorted(random.sample(range(taille), 2))
    
    enfant = [-1] * taille
    enfant[point1:point2] = parent1[point1:point2]
    
    mapping = {}
    for i in range(point1, point2):
        mapping[parent1[i]] = parent2[i]
    
    for i in list(range(point1)) + list(range(point2, taille)):
        candidat = parent2[i]
        while candidat in enfant:
            candidat = mapping.get(candidat, candidat)
            if candidat not in mapping:
                break
        enfant[i] = candidat
    
    return enfant


def mutation_echange(individu, taux_mutation):
    """Mutation par échange de deux villes"""
    if random.random() < taux_mutation:
        i, j = random.sample(range(len(individu)), 2)
        individu[i], individu[j] = individu[j], individu[i]
    return individu


def algorithme_genetique(matrice_distances, taille_population, nombre_generations,
                        taux_mutation, taux_croisement, elitisme=True):
    """
    Algorithme Génétique pour le problème du voyageur de commerce
    
    Paramètres:
    - matrice_distances: matrice des distances entre villes
    - taille_population: nombre d'individus dans la population
    - nombre_generations: nombre de générations à exécuter
    - taux_mutation: probabilité de mutation
    - taux_croisement: probabilité de croisement
    - elitisme: si True, garde le meilleur individu à chaque génération
    """
    
    nombre_villes = len(matrice_distances)
    population = creer_population_initiale(taille_population, nombre_villes)
    
    meilleure_solution = None
    meilleure_distance = float('inf')
    
    historique_distances = []
    historique_meilleures = []
    
    print(f"{'='*60}")
    print(f"Démarrage de l'Algorithme Génétique")
    print(f"Méthode de sélection: Rang (basée sur le classement)")
    print(f"{'='*60}")
    print(f"Taille de la population: {taille_population}")
    print(f"Nombre de générations: {nombre_generations}")
    print(f"Taux de mutation: {taux_mutation}")
    print(f"Taux de croisement: {taux_croisement}")
    print(f"{'='*60}\n")
    
    for generation in range(nombre_generations):
        fitnesses = [calculer_fitness(ind, matrice_distances) for ind in population]
        
        meilleur_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        distance_generation = calculer_distance_totale(population[meilleur_idx], matrice_distances)
        
        if distance_generation < meilleure_distance:
            meilleure_distance = distance_generation
            meilleure_solution = population[meilleur_idx][:]
            print(f"Génération {generation}: Nouvelle meilleure solution trouvée!")
            print(f"  Distance: {meilleure_distance}")
        
        historique_distances.append(distance_generation)
        historique_meilleures.append(meilleure_distance)
        
        if generation % 100 == 0:
            distance_moyenne = sum(1/f for f in fitnesses) / len(fitnesses)
            print(f"Génération {generation}: "
                  f"Meilleure={meilleure_distance}, "
                  f"Génération actuelle={distance_generation:.2f}, "
                  f"Moyenne={distance_moyenne:.2f}")
        
        nouvelle_population = []
        
        if elitisme:
            nouvelle_population.append(population[meilleur_idx][:])
        
        while len(nouvelle_population) < taille_population:
            parent1 = selection_rang(population, fitnesses)
            parent2 = selection_rang(population, fitnesses)
            
            if random.random() < taux_croisement:
                enfant = croisement_pmx(parent1, parent2)
            else:
                enfant = parent1[:]
            
            enfant = mutation_echange(enfant, taux_mutation)
            
            nouvelle_population.append(enfant)
        
        population = nouvelle_population
    
    print(f"\n{'='*60}")
    print(f"Fin de l'Algorithme Génétique")
    print(f"{'='*60}")
    print(f"Nombre total de générations: {nombre_generations}")
    
    return meilleure_solution, meilleure_distance, historique_distances, historique_meilleures


matrice_distances = [
    [0, 2, 2, 7, 15, 2, 5, 7, 6, 5],
    [2, 0, 10, 4, 7, 3, 7, 15, 8, 2],
    [2, 10, 0, 1, 4, 3, 3, 4, 2, 3],
    [7, 4, 1, 0, 2, 15, 7, 7, 5, 4],
    [7, 10, 4, 2, 0, 7, 3, 2, 2, 7],
    [2, 3, 3, 7, 7, 0, 1, 7, 2, 10],
    [5, 7, 3, 7, 3, 1, 0, 2, 1, 3],
    [7, 7, 4, 7, 2, 7, 2, 0, 1, 10],
    [6, 8, 2, 5, 2, 2, 1, 1, 0, 15],
    [5, 2, 3, 4, 7, 10, 3, 10, 15, 0]
]

taille_population = 100
nombre_generations = 500
taux_mutation = 0.2
taux_croisement = 0.8

meilleure_solution, meilleure_distance, historique_distances, historique_meilleures = \
    algorithme_genetique(matrice_distances,
                        taille_population,
                        nombre_generations,
                        taux_mutation,
                        taux_croisement,
                        elitisme=True)

print(f"\n{'='*60}")
print(f"RÉSULTAT FINAL")
print(f"{'='*60}")
print(f"Meilleure solution trouvée (Algorithme Génétique): {meilleure_solution}")
print(f"Distance minimale: {meilleure_distance}")
print(f"{'='*60}")