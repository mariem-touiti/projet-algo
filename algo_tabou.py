import streamlit as st
import random
from collections import deque
import matplotlib.pyplot as plt

# === Fonctions de base ===

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale d'une solution (route complète)"""
    distance_totale = 0
    for i in range(len(solution) - 1):
        distance_totale += matrice_distances[solution[i]][solution[i + 1]]
    distance_totale += matrice_distances[solution[-1]][solution[0]]
    return distance_totale


def generer_voisinage(solution):
    """Génère tous les voisins possibles par échange de deux villes (2-opt)"""
    voisins = []
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            voisin = solution[:]
            voisin[i], voisin[j] = voisin[j], voisin[i]
            voisins.append((voisin, (i, j)))
    return voisins


def est_tabou(mouvement, liste_tabou):
    """Vérifie si un mouvement est dans la liste tabou"""
    return mouvement in liste_tabou or (mouvement[1], mouvement[0]) in liste_tabou


def recherche_tabou(matrice_distances, taille_liste_tabou, nombre_iterations, taille_voisinage_max=50):
    nombre_villes = len(matrice_distances)
    solution_actuelle = list(range(nombre_villes))
    random.shuffle(solution_actuelle)
    distance_actuelle = calculer_distance_totale(solution_actuelle, matrice_distances)
    
    meilleure_solution = solution_actuelle[:]
    meilleure_distance = distance_actuelle
    liste_tabou = deque(maxlen=taille_liste_tabou)
    
    historique_distances = []
    historique_meilleures = []
    iterations_sans_amelioration = 0

    for iteration in range(nombre_iterations):
        voisins = generer_voisinage(solution_actuelle)
        if len(voisins) > taille_voisinage_max:
            voisins = random.sample(voisins, taille_voisinage_max)

        meilleur_voisin = None
        meilleur_mouvement = None
        meilleure_distance_voisin = float('inf')
        meilleur_voisin_tabou = None
        meilleure_distance_tabou = float('inf')

        for voisin, mouvement in voisins:
            distance_voisin = calculer_distance_totale(voisin, matrice_distances)
            if not est_tabou(mouvement, liste_tabou):
                if distance_voisin < meilleure_distance_voisin:
                    meilleure_distance_voisin = distance_voisin
                    meilleur_voisin = voisin
                    meilleur_mouvement = mouvement
            else:
                if distance_voisin < meilleure_distance_tabou:
                    meilleure_distance_tabou = distance_voisin
                    meilleur_voisin_tabou = voisin
                    mouvement_tabou = mouvement

        if meilleure_distance_tabou < meilleure_distance:
            solution_actuelle = meilleur_voisin_tabou
            distance_actuelle = meilleure_distance_tabou
            liste_tabou.append(mouvement_tabou)
        elif meilleur_voisin is not None:
            solution_actuelle = meilleur_voisin
            distance_actuelle = meilleure_distance_voisin
            liste_tabou.append(meilleur_mouvement)
        else:
            break

        if distance_actuelle < meilleure_distance:
            meilleure_solution = solution_actuelle[:]
            meilleure_distance = distance_actuelle
            iterations_sans_amelioration = 0
        else:
            iterations_sans_amelioration += 1

        historique_distances.append(distance_actuelle)
        historique_meilleures.append(meilleure_distance)

        if iterations_sans_amelioration > nombre_iterations // 5:
            break

    return meilleure_solution, meilleure_distance, historique_distances, historique_meilleures


# === Interface Streamlit ===

st.title("🔍 Recherche Tabou – Problème du Voyageur de Commerce (TSP)")

st.sidebar.header("⚙️ Paramètres de l'algorithme")
taille_liste_tabou = st.sidebar.slider("Taille de la liste tabou", 5, 100, 20)
nombre_iterations = st.sidebar.slider("Nombre d'itérations", 100, 5000, 1000)
taille_voisinage_max = st.sidebar.slider("Taille max du voisinage", 10, 200, 50)

st.write("### 🗺️ Matrice de distances utilisée")

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

if st.button("🚀 Lancer la recherche tabou"):
    st.write("### ⏳ Exécution en cours...")
    meilleure_solution, meilleure_distance, historique_distances, historique_meilleures = recherche_tabou(
        matrice_distances,
        taille_liste_tabou,
        nombre_iterations,
        taille_voisinage_max
    )

    st.success("✅ Recherche terminée !")

    st.write("### 📈 Résultats")
    st.write(f"**Meilleure distance trouvée :** {meilleure_distance}")
    st.write(f"**Meilleure solution (ordre des villes) :** {meilleure_solution}")

    # === Graphiques ===
    fig, ax = plt.subplots()
    ax.plot(historique_distances, label="Distance actuelle")
    ax.plot(historique_meilleures, label="Meilleure distance", linestyle='--')
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Distance")
    ax.set_title("Évolution des distances pendant la recherche tabou")
    ax.legend()
    st.pyplot(fig)

    st.write("### 🧭 Détails supplémentaires")
    st.write(f"**Taille de la liste tabou :** {taille_liste_tabou}")
    st.write(f"**Nombre d’itérations exécutées :** {len(historique_distances)}")

st.info("💡 Ajuste les paramètres dans le menu à gauche, puis clique sur **Lancer la recherche tabou**.")
