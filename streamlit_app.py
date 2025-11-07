import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import time

# ===============================
#  🔹 Fonctions utilitaires
# ===============================
def distance(v1, v2):
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def generer_matrice(villes):
    n = len(villes)
    matrice = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrice[i][j] = int(distance(villes[i], villes[j]))
    return matrice

def calcul_energie(etat, matrice_energie):
    energie = 0
    for i in range(len(etat) - 1):
        energie += matrice_energie[etat[i]][etat[i+1]]
    energie += matrice_energie[etat[-1]][etat[0]]
    return energie

def generer_voisin(etat):
    voisin = etat[:]
    i, j = random.sample(range(len(etat)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

# ===============================
#  🔥 Recuit simulé
# ===============================
def recuit_simule(matrice, temp_initiale=1000, refroid=0.98, iterations=500):
    etat = list(range(len(matrice)))
    random.shuffle(etat)
    energie = calcul_energie(etat, matrice)
    meilleur, best_energie = etat[:], energie
    temp = temp_initiale

    for _ in range(iterations):
        voisin = generer_voisin(etat)
        energie_voisin = calcul_energie(voisin, matrice)
        delta = energie_voisin - energie
        if delta < 0 or random.random() < math.exp(-delta / temp):
            etat, energie = voisin, energie_voisin
        if energie < best_energie:
            meilleur, best_energie = etat[:], energie
        temp *= refroid
    return meilleur, best_energie

# ===============================
#  🚫 Recherche Tabou
# ===============================
def tabu_search(matrice, iterations=200, tabu_size=15):
    n = len(matrice)
    current = random.sample(range(n), n)
    best = current[:]
    best_cost = calcul_energie(current, matrice)
    tabu_list = []

    for _ in range(iterations):
        voisins = [generer_voisin(current) for _ in range(30)]
        candidats = [(v, calcul_energie(v, matrice)) for v in voisins if v not in tabu_list]
        if not candidats:
            continue
        candidats.sort(key=lambda x: x[1])
        meilleur_voisin, cout = candidats[0]

        if cout < best_cost:
            best, best_cost = meilleur_voisin[:], cout

        tabu_list.append(meilleur_voisin)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        current = meilleur_voisin[:]
    return best, best_cost

# ===============================
#  🧬 Algorithme Génétique
# ===============================
def algo_genetique(matrice, population_size=50, generations=100, mutation_rate=0.1):
    def crossover(p1, p2):
        point = random.randint(1, len(p1)-2)
        child = p1[:point] + [x for x in p2 if x not in p1[:point]]
        return child

    def mutate(etat):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(etat)), 2)
            etat[i], etat[j] = etat[j], etat[i]
        return etat

    n = len(matrice)
    population = [random.sample(range(n), n) for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=lambda x: calcul_energie(x, matrice))
        new_pop = population[:10]
        while len(new_pop) < population_size:
            p1, p2 = random.sample(population[:20], 2)
            new_pop.append(mutate(crossover(p1, p2)))
        population = new_pop
    best = min(population, key=lambda x: calcul_energie(x, matrice))
    return best, calcul_energie(best, matrice)

# ===============================
#  🎨 Visualisation
# ===============================
def plot_villes(villes, chemin):
    fig, ax = plt.subplots(facecolor="#ffe6f0")  # 🎨 Fond rose clair

    # Coordonnées du chemin
    x = [villes[i][0] for i in chemin]
    y = [villes[i][1] for i in chemin]

    # ➤ Tracé du trajet en mauve
    ax.plot(x, y, color="#b266ff", marker='o', markersize=6, linewidth=2.5, label='Trajet optimal')

    # Labels des villes
    for i, (xv, yv) in enumerate(villes):
        ax.text(xv + 0.3, yv + 0.3, f"Ville {i}", fontsize=9, color="#800080")  # violet foncé

    # 🔹 Départ en rose foncé
    ax.scatter(villes[chemin[0]][0], villes[chemin[0]][1],
               color='#ff4da6', s=120, edgecolors='black', label='Départ')

    # 🔹 Arrivée en violet foncé
    ax.scatter(villes[chemin[-1]][0], villes[chemin[-1]][1],
               color='#9933ff', s=120, edgecolors='black', label='Arrivée')

    # Autres réglages esthétiques
    ax.legend()
    ax.set_title("💜 Chemin optimal (trajet non cyclique)", color="#660066", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor("#ffe6f0")  # fond intérieur du graphique rose clair

    return fig


# ===============================
#  🌍 Interface Streamlit
# ===============================
st.title(" Optimisation combinatoire - Visualisation des algorithmes")
st.markdown("""
Choisissez un *algorithme d’optimisation* et observez comment il trouve le chemin optimal entre plusieurs villes.
""")

algo = st.selectbox("Choisissez un algorithme :", ["Recuit simulé", "Recherche Tabou", "Algorithme Génétique"])
nb_villes = st.slider("Nombre de villes :", 4, 15)

# Génération des coordonnées
villes = [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(nb_villes)]
matrice = generer_matrice(villes)

st.write("### 🧮 Matrice des distances (arrondie)")
st.dataframe([[round(x, 2) for x in row] for row in matrice])

if st.button("🚀 Lancer l’algorithme"):
    # --- Démarrer le chronomètre ---
    start_time = time.time()

    # --- Exécution de l'algorithme choisi ---
    if algo == "Recuit simulé":
        chemin, cout = recuit_simule(matrice)
    elif algo == "Recherche Tabou":
        chemin, cout = tabu_search(matrice)
    else:
        chemin, cout = algo_genetique(matrice)

    # --- Arrêter le chronomètre ---
    end_time = time.time()
    execution_time = end_time - start_time

    # --- Résultats ---
    st.success(f"✅ Chemin trouvé : {chemin}")
    st.info(f"⏱ Temps d'exécution : {execution_time:.4f} secondes")

    # --- Visualisation ---
    fig = plot_villes(villes, chemin)
    st.pyplot(fig)


# ===============================
# 🎀 STYLE GÉNÉRAL DE LA PAGE
# ===============================
st.markdown("""
    <style>
    /* 🌸 Dégradé de fond pour toute la page */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #ffe6f0, #f3d1ff);
    }

    /* 🎨 Titres principaux */
    h1 {
        color: #a64ca6 !important;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: 800;
        text-shadow: 1px 1px 3px #ffffff;
    }

    h2, h3 {
        color: #b266ff !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    /* ✨ Boutons */
    div.stButton > button {
        background-color: #cc66ff;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #ff66b2;
        color: #fff;
        transform: scale(1.05);
    }

    /* 🩷 Encadré des DataFrames */
    .stDataFrame {
        border: 2px solid #cc99ff;
        border-radius: 10px;
        background-color: #fff8ff;
    }

    /* 💬 Texte */
    p, label {
        color: #4a004a !important;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)











