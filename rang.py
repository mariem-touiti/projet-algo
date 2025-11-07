import streamlit as st
import random
import math
import matplotlib.pyplot as plt

# -------------------------
# Fonctions principales
# -------------------------

def calculer_distance_totale(solution, matrice_distances):
    distance_totale = 0
    for i in range(len(solution) - 1):
        distance_totale += matrice_distances[solution[i]][solution[i + 1]]
    distance_totale += matrice_distances[solution[-1]][solution[0]]
    return distance_totale


def creer_population_initiale(taille_population, nombre_villes):
    population = []
    for _ in range(taille_population):
        individu = list(range(nombre_villes))
        random.shuffle(individu)
        population.append(individu)
    return population


def calculer_fitness(individu, matrice_distances):
    distance = calculer_distance_totale(individu, matrice_distances)
    return 1 / distance if distance > 0 else 0


def selection_rang(population, fitnesses):
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
    if random.random() < taux_mutation:
        i, j = random.sample(range(len(individu)), 2)
        individu[i], individu[j] = individu[j], individu[i]
    return individu


def algorithme_genetique(matrice_distances, taille_population, nombre_generations,
                         taux_mutation, taux_croisement, elitisme=True, progress=None):

    nombre_villes = len(matrice_distances)
    population = creer_population_initiale(taille_population, nombre_villes)
    meilleure_solution = None
    meilleure_distance = float('inf')
    historique_distances = []
    historique_meilleures = []

    for generation in range(nombre_generations):
        fitnesses = [calculer_fitness(ind, matrice_distances) for ind in population]
        meilleur_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        distance_generation = calculer_distance_totale(population[meilleur_idx], matrice_distances)

        if distance_generation < meilleure_distance:
            meilleure_distance = distance_generation
            meilleure_solution = population[meilleur_idx][:]

        historique_distances.append(distance_generation)
        historique_meilleures.append(meilleure_distance)

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

        if progress:
            progress.progress((generation + 1) / nombre_generations)

    return meilleure_solution, meilleure_distance, historique_distances, historique_meilleures


# -------------------------
# Interface Streamlit
# -------------------------

st.title("🧬 Algorithme Génétique - Voyageur de Commerce (TSP)")
st.write("Optimisation du chemin entre plusieurs villes à l’aide d’un algorithme génétique.")

# Paramètres interactifs
taille_population = st.slider("Taille de la population", 10, 200, 100, 10)
nombre_generations = st.slider("Nombre de générations", 50, 1000, 500, 50)
taux_mutation = st.slider("Taux de mutation", 0.01, 1.0, 0.2)
taux_croisement = st.slider("Taux de croisement", 0.1, 1.0, 0.8)
elitisme = st.checkbox("Activer l’élitisme", True)

# Matrice exemple
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

# Bouton d’exécution
if st.button("🚀 Lancer l’algorithme"):
    with st.spinner("Exécution en cours..."):
        progress = st.progress(0)
        meilleure_solution, meilleure_distance, historique_distances, historique_meilleures = \
            algorithme_genetique(matrice_distances, taille_population, nombre_generations,
                                 taux_mutation, taux_croisement, elitisme, progress)

    st.success("✅ Exécution terminée !")

    st.subheader("📊 Résultats")
    st.write(f"**Meilleure solution trouvée :** {meilleure_solution}")
    st.write(f"**Distance minimale :** {meilleure_distance:.2f}")

    # Graphique de convergence
    fig, ax = plt.subplots()
    ax.plot(historique_distances, label="Distance actuelle")
    ax.plot(historique_meilleures, label="Meilleure distance", linestyle="--")
    ax.set_title("Évolution de la distance au fil des générations")
    ax.set_xlabel("Générations")
    ax.set_ylabel("Distance")
    ax.legend()
    st.pyplot(fig)

    # Visualisation du chemin
    st.subheader("🗺️ Visualisation du chemin trouvé")
    coords = [(random.random(), random.random()) for _ in range(len(matrice_distances))]
    fig2, ax2 = plt.subplots()
    for i in range(len(meilleure_solution)):
        x1, y1 = coords[meilleure_solution[i]]
        x2, y2 = coords[meilleure_solution[(i+1) % len(meilleure_solution)]]
        ax2.plot([x1, x2], [y1, y2], 'bo-')
        ax2.text(x1, y1, str(meilleure_solution[i]), fontsize=9, color='red')
    ax2.set_title("Chemin optimal (visualisation aléatoire des villes)")
    st.pyplot(fig2)
