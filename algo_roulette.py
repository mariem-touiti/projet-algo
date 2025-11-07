import streamlit as st
import random
import matplotlib.pyplot as plt

# ======================
# Fonctions de base
# ======================

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


def selection_tournoi(population, fitnesses, taille_tournoi=3):
    """Sélection par tournoi: choisit le meilleur parmi k individus aléatoires"""
    indices = random.sample(range(len(population)), taille_tournoi)
    meilleur_idx = max(indices, key=lambda i: fitnesses[i])
    return population[meilleur_idx][:]


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
            parent1 = selection_tournoi(population, fitnesses)
            parent2 = selection_tournoi(population, fitnesses)
            
            if random.random() < taux_croisement:
                enfant = croisement_pmx(parent1, parent2)
            else:
                enfant = parent1[:]
            
            enfant = mutation_echange(enfant, taux_mutation)
            nouvelle_population.append(enfant)
        
        population = nouvelle_population
    
    return meilleure_solution, meilleure_distance, historique_distances, historique_meilleures


# ======================
# Interface Streamlit
# ======================

st.title("🧬 Algorithme Génétique - Problème du Voyageur de Commerce (TSP)")

st.sidebar.header("⚙️ Paramètres de l'algorithme")

taille_population = st.sidebar.slider("Taille de la population", 10, 300, 100)
nombre_generations = st.sidebar.slider("Nombre de générations", 100, 2000, 500)
taux_mutation = st.sidebar.slider("Taux de mutation", 0.0, 1.0, 0.2)
taux_croisement = st.sidebar.slider("Taux de croisement", 0.0, 1.0, 0.8)
elitisme = st.sidebar.checkbox("Activer l'élitisme", True)

st.write("### 🗺️ Matrice de distances (entre 10 villes)")

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

st.table(matrice_distances)

if st.button("🚀 Lancer l'algorithme génétique"):
    st.write("### ⏳ Exécution en cours...")

    meilleure_solution, meilleure_distance, historique_distances, historique_meilleures = \
        algorithme_genetique(matrice_distances,
                            taille_population,
                            nombre_generations,
                            taux_mutation,
                            taux_croisement,
                            elitisme)

    st.success("✅ Exécution terminée !")

    st.write("### 📊 Résultats")
    st.write(f"**Meilleure distance trouvée :** {meilleure_distance}")
    st.write(f"**Ordre des villes :** {meilleure_solution}")

    # === Graphique évolution ===
    fig, ax = plt.subplots()
    ax.plot(historique_distances, label="Distance génération")
    ax.plot(historique_meilleures, label="Meilleure distance", linestyle="--")
    ax.set_xlabel("Générations")
    ax.set_ylabel("Distance")
    ax.set_title("Évolution des distances")
    ax.legend()
    st.pyplot(fig)

    # === Visualisation du trajet ===
    st.write("### 🧭 Visualisation du chemin optimal")
    nb_villes = len(meilleure_solution)
    coords = {i: (random.random(), random.random()) for i in range(nb_villes)}

    fig2, ax2 = plt.subplots()
    for i in range(nb_villes):
        ville_actuelle = meilleure_solution[i]
        ville_suivante = meilleure_solution[(i + 1) % nb_villes]
        x1, y1 = coords[ville_actuelle]
        x2, y2 = coords[ville_suivante]
        ax2.plot([x1, x2], [y1, y2], "bo-")
        ax2.text(x1, y1, str(ville_actuelle), fontsize=10, color="red")
    ax2.set_title("Chemin optimal entre les villes (positions aléatoires)")
    st.pyplot(fig2)

st.info("💡 Ajuste les paramètres dans la barre latérale, puis clique sur **Lancer l'algorithme génétique**.")
