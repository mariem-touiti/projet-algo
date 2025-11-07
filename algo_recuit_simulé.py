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


def generer_voisin(solution):
    voisin = solution[:]
    i, j = random.sample(range(len(solution)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin


def recuit_simule(matrice_distances, temperature_initiale, temperature_finale, 
                  taux_refroidissement, iterations_par_temperature, progress_bar=None):
    
    nombre_villes = len(matrice_distances)
    solution_actuelle = list(range(nombre_villes))
    random.shuffle(solution_actuelle)
    
    meilleure_solution = solution_actuelle[:]
    meilleure_distance = calculer_distance_totale(solution_actuelle, matrice_distances)
    distance_actuelle = meilleure_distance
    temperature = temperature_initiale
    iteration_globale = 0
    historique_distances = []
    historique_temperatures = []
    
    while temperature > temperature_finale:
        for _ in range(iterations_par_temperature):
            iteration_globale += 1
            
            solution_voisine = generer_voisin(solution_actuelle)
            distance_voisine = calculer_distance_totale(solution_voisine, matrice_distances)
            delta = distance_voisine - distance_actuelle
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                solution_actuelle = solution_voisine
                distance_actuelle = distance_voisine
                
                if distance_actuelle < meilleure_distance:
                    meilleure_solution = solution_actuelle[:]
                    meilleure_distance = distance_actuelle
            
            historique_distances.append(distance_actuelle)
            historique_temperatures.append(temperature)
        
        temperature *= taux_refroidissement
        
        if progress_bar:
            progress_bar.progress(min(1.0, temperature_initiale / max(temperature, 1)))
    
    return meilleure_solution, meilleure_distance, historique_distances, historique_temperatures


# -------------------------
# Interface Streamlit
# -------------------------

st.title("🚀 Algorithme du Recuit Simulé - Problème du Voyageur de Commerce (TSP)")

# Paramètres interactifs
temperature_initiale = st.slider("Température initiale", 100, 5000, 1000, 100)
temperature_finale = st.slider("Température finale", 0.001, 1.0, 0.01)
taux_refroidissement = st.slider("Taux de refroidissement", 0.80, 0.999, 0.95)
iterations_par_temperature = st.slider("Itérations par température", 10, 500, 100, 10)

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

if st.button("🎯 Lancer le recuit simulé"):
    with st.spinner("Exécution de l’algorithme..."):
        progress_bar = st.progress(0)
        meilleure_solution, meilleure_distance, historique_distances, historique_temperatures = recuit_simule(
            matrice_distances,
            temperature_initiale,
            temperature_finale,
            taux_refroidissement,
            iterations_par_temperature,
            progress_bar
        )
    
    st.success("✅ Exécution terminée !")

    st.subheader("📊 Résultats finaux")
    st.write(f"**Meilleure solution trouvée :** {meilleure_solution}")
    st.write(f"**Distance minimale :** {meilleure_distance}")

    # Graphique de convergence
    fig, ax = plt.subplots()
    ax.plot(historique_distances, label="Distance actuelle")
    ax.set_title("Évolution de la distance au fil des itérations")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Distance")
    ax.legend()
    st.pyplot(fig)

    # Graphique des températures
    fig2, ax2 = plt.subplots()
    ax2.plot(historique_temperatures, label="Température", color="orange")
    ax2.set_title("Évolution de la température")
    ax2.set_xlabel("Itérations")
    ax2.set_ylabel("Température")
    ax2.legend()
    st.pyplot(fig2)

    # Visualisation du chemin
    st.subheader("🗺️ Visualisation du chemin trouvé")
    coords = [(random.random(), random.random()) for _ in range(len(matrice_distances))]
    fig3, ax3 = plt.subplots()
    for i in range(len(meilleure_solution)):
        x1, y1 = coords[meilleure_solution[i]]
        x2, y2 = coords[meilleure_solution[(i+1) % len(meilleure_solution)]]
        ax3.plot([x1, x2], [y1, y2], 'bo-')
    ax3.set_title("Chemin optimal (approximation visuelle)")
    st.pyplot(fig3)
