# TP1 IA

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, bernoulli, binom


# LOI UNIFORME
 

faces = np.arange(1, 7)
X = randint(1, 7)

#a
N = 100
lancers = X.rvs(size=N)

# b
effectifs = np.array([np.sum(lancers == k) for k in faces])
frequences = effectifs / N

print("resultat pour 100:")
for k, eff, freq in zip(faces, effectifs, frequences):
  print(f"Face {k} : {eff} apparitions, fréquence = {freq:.3f}")

# c
plt.figure(figsize=(8, 4))
plt.bar(
    faces,
    frequences,
    width=0.5,
    color="skyblue",
    edgecolor="black",
    label=f"fréquences expé (N={N})",
)
plt.xlabel("face du dé (k)")
plt.ylabel("frequence")
plt.title(f"fréquences d'apparition des faces (N = {N})")
plt.xticks(faces)
plt.ylim(0, max(frequences) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

# d
pmf = X.pmf(faces)

plt.figure(figsize=(8, 4))
plt.bar(
    faces - 0.15,
    frequences,
    width=0.3,
    color="skyblue",
    edgecolor="black",
    label=f"Fréquences exp. (N={N})",
)
plt.bar(
    faces + 0.15,
    pmf,
    width=0.3,
    color="salmon",
    edgecolor="black",
    label="fonction de masse f_X(k) = 1/6",
)
plt.xlabel("Face du dé (k)")
plt.ylabel("probabilité / frequence")
plt.title("comparaison : frequences expé vs fonction de masse")
plt.xticks(faces)
plt.ylim(0, max(max(frequences), 1 / 6) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

#e
k_vals = np.arange(0, 8)
cdf = X.cdf(k_vals)

plt.figure(figsize=(8, 4))
plt.step(k_vals, cdf, where="post", color="darkgreen", linewidth=2)
plt.scatter(faces, X.cdf(faces), color="darkgreen", zorder=5)
plt.xlabel("k")
plt.ylabel("F_X(k) = P(X <= k)")
plt.title("fonc de répartition f_x(k) de la loi uniforme")
plt.xticks(k_vals)
plt.ylim(-0.05, 1.1)
plt.grid(True, linestyle="--", alpha=0.6)

#f
p_inf_egal_2 = X.cdf(2)
print(f"\n(f) P(X <= 2) = {p_inf_egal_2:.4f} (soit 2/6 = 1/3)")

# g
p_sup_3 = 1 - X.cdf(3)  
p_sup_4 = 1 - X.cdf(4)  
print(f"(g) P(X > 3) = {p_sup_3:.4f} (soit 3/6 = 1/2)")
print(f"P(X > 4) = {p_sup_4:.4f} (soit 2/6 = 1/3)")

# h
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, N_val in zip(axes, [10, 1000]):
  #a
  sim = X.rvs(size=N_val)

  #b
  eff = np.array([np.sum(sim == k) for k in faces])
  freq = eff / N_val

  print(f"\npour N = {N_val}")
  for k, e, f in zip(faces, eff, freq):
    print(f"face {k} : {e} apparitions,fréquence = {f:.3f}")

  #c
  ax.bar(
      faces,
      freq,
      width=0.5,
      color="cornflowerblue",
      edgecolor="black",
      label="fréquences expé",
  )
  ax.axhline(
      1 / 6, color="red", linestyle="--", label="probabilité theorique (1/6)"
  )
  ax.set_title(f"N = {N_val} lancers")
  ax.set_xlabel("face du dé (k)")
  ax.set_xticks(faces)
  ax.grid(axis="y", linestyle="--", alpha=0.6)
  ax.legend()

axes[0].set_ylabel("fréquence")
plt.suptitle("convergence des frequences vers la loi théorique")
plt.tight_layout()
#plt.show()

# LOIS DE BERNOULLI ET BINOMIALE

p = 0.5
X = bernoulli(p)
valeurs_X = np.array([0, 1])

#a
N = 100
reponses_X = X.rvs(size=N)

# b
effectifs_X = np.array([np.sum(reponses_X == k) for k in valeurs_X])
frequences_X = effectifs_X / N

print("bernoulli (N = 100 candidats, 1 question)")
print(
    f"réponses incorrectes (0): {effectifs_X[0]} et fréquence ="
    f" {frequences_X[0]:.3f}"
)
print(
    f"réponses correctes   (1) : {effectifs_X[1]} et fréquence ="
    f" {frequences_X[1]:.3f}"
)

#c
plt.figure(figsize=(7, 4))
plt.bar(
    valeurs_X,
    frequences_X,
    width=0.4,
    color="skyblue",
    edgecolor="black",
    label=f"freq expé (N={N})",
)
plt.xlabel("issue k (0 = incorrecte, 1 = correcte)")
plt.ylabel("fréquence")
plt.title("loi bernoulli : frequences des réponses (N = 100)")
plt.xticks(valeurs_X)
plt.ylim(0, max(frequences_X) + 0.15)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

#d
pmf_X = X.pmf(valeurs_X)

plt.figure(figsize=(7, 4))
plt.bar(
    valeurs_X - 0.12,
    frequences_X,
    width=0.24,
    color="skyblue",
    edgecolor="black",
    label=f"frequence expé (N={N})",
)
plt.bar(
    valeurs_X + 0.12,
    pmf_X,
    width=0.24,
    color="salmon",
    edgecolor="black",
    label="fonction de masse f_X(k) = 0.5",
)
plt.xlabel("issue k (0 = Incorrecte, 1 = Correcte)")
plt.ylabel("proba / freq")
plt.title("loi de bernoulli : frequences experi vs fonction de masse")
plt.xticks(valeurs_X)
plt.ylim(0, max(max(frequences_X), 0.5) + 0.15)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

#partie 2 de lois de bernoulli et binomiale

n = 10
Y = binom(n, p)
valeurs_Y = np.arange(0, n + 1)  

#a
N = 100
scores_Y = Y.rvs(size=N)

#b
effectifs_Y = np.array([np.sum(scores_Y == k) for k in valeurs_Y])
frequences_Y = effectifs_Y / N

print("\n loi Binomiale (N = 100 candidats, n = 10 questions)")
for k, eff, freq in zip(valeurs_Y, effectifs_Y, frequences_Y):
  print(f"{k:2d} bonnes réponses : {eff:2d} candidats, fréquence = {freq:.3f}")

#c
pmf_Y = Y.pmf(valeurs_Y)

plt.figure(figsize=(8, 4))
plt.bar(
    valeurs_Y,
    pmf_Y,
    width=0.5,
    color="salmon",
    edgecolor="black",
    label="f_Y(k) = P(Y = k)",
)
plt.xlabel("nb de reponses correctes (k)")
plt.ylabel("proba")
plt.title("fonc de masse de la loi binomiale B(10, 0.5)")
plt.xticks(valeurs_Y)
plt.ylim(0, max(pmf_Y) + 0.05)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

#d
p_exact_2 = Y.pmf(2)
print(f"\n(d) P(Y = 2) = {p_exact_2:.4f}")

#e
cdf_Y = Y.cdf(valeurs_Y)

plt.figure(figsize=(8, 4))
plt.step(
    valeurs_Y,
    cdf_Y,
    where="post",
    color="darkgreen",
    linewidth=2,
    label="F_Y(k) = P(Y <= k)",
)
plt.scatter(valeurs_Y, cdf_Y, color="darkgreen", zorder=5)
plt.xlabel("nb de reponses correctes (k)")
plt.ylabel("proba cumulée")
plt.title("fonction de répartition F_Y(k) de la loi binomiale B(10, 0.5)")
plt.xticks(valeurs_Y)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

#f
p_au_moins_5 = 1 - Y.cdf(4)
print(f"(f) P(Y >= 5) = {p_au_moins_5:.4f}")

#g
n_vals = np.arange(5, 21)  
probs_au_moins_5 = [1 - binom.cdf(4, n_i, p) for n_i in n_vals]

plt.figure(figsize=(9, 4))
plt.plot(
    n_vals,
    probs_au_moins_5,
    marker="o",
    linestyle="-",
    color="purple",
    linewidth=2,
)
plt.xlabel("nb de questions du questionnaire (n)")
plt.ylabel("P(Y >= 5)")
plt.title("zvolution de P(Y >= 5) en fonction du nb de questions n")
plt.xticks(n_vals)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()