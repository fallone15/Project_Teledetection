# 🌲 Projet Télédétection Sentinel-2 : Détection du Dépérissement Forestier (Vosges)
## Synthèse Méthodologique Complète : Régression Harmonique & Étapes 5, 6 et 7

Ce document regroupe l'ensemble des explications théoriques, mathématiques et informatiques du projet. Il est conçu pour servir de support d'étude et de fiche de préparation à la soutenance devant votre professeur.

---

## 📌 Table des matières
1. [Contexte et Logique d'Ensemble](#1-contexte-et-logique-densemble)
2. [Étape 3 : Régression Harmonique (Calibrage 2016)](#2-étape-3--régression-harmonique-calibrage-2016)
3. [Étape 5 : Méthode de Détection du Dépérissement (2018)](#3-étape-5--méthode-de-détection-du-dépérissement-2018)
4. [Étape 6 : Discrimination Coupes Rases vs Scolytes](#4-étape-6--discrimination-coupes-rases-vs-scolytes)
5. [Étape 7 : Discrimination Sécheresse vs Scolytes](#5-étape-7--discrimination-sécheresse-vs-scolytes)
6. [Fiche Mémo / Discours pour la Présentation Orale](#6-fiche-mémo--discours-pour-la-présentation-orale)

---

## 1. Contexte et Logique d'Ensemble

### Le contexte local et biologique (Bassin d'Épinal / Massif Vosgien)
- **Localisation de la zone d'étude :** Les coordonnées de notre zone d'intérêt (`47.91°N - 47.94°N, 6.55°E - 6.57°E`) se situent à environ 30 km au sud d'**Épinal**, dans le massif des Vosges (secteur de la haute vallée de la Moselle / versant vosgien entre Remiremont et Rupt-sur-Moselle).
- **Enjeu territorial & filière bois :** Épinal et les Vosges constituent le cœur historique de la filière forêt-bois en France (scieries, ONF, ENSTIB). Les peuplements d'épicéas communs (*Picea abies*), plantés massivement après-guerre sur des stations parfois trop sèches en basse et moyenne altitude, ont subi un choc hydrique extrême lors de la canicule de 2018.
- **La crise des scolytes :** L'affaiblissement hydrique a ouvert la voie à une prolifération épidémique de **scolytes typographes** (*Ips typographus*). Ces insectes sous-corticaux creusent des galeries de ponte dans le liber, interrompent les flux de sève élaborée et provoquent la mort fulgurante de peuplements entiers sur pied. L'enjeu pour les gestionnaires forestiers et les scieries vosgiennes était de cartographier au plus vite les foyers pour exploiter les bois avant dégradation (bleuissement fongique).

### L'indice spectral clé : CRSWIR (*Continuum Removal SWIR*)
Alors que le NDVI mesure la densité de chlorophylle (structure du couvert), la méthode standard (notamment issue du projet **FORDEAD**) utilise le **CRSWIR** :
$$\text{CRSWIR} = \frac{B_{11}}{B_{8A} + \frac{B_{12} - B_{8A}}{\lambda_{12} - \lambda_{8A}} \times (\lambda_{11} - \lambda_{8A})}$$
avec les longueurs d'ondes Sentinel-2 : $\lambda_{8A} = 864\,\text{nm}$, $\lambda_{11} = 1610.4\,\text{nm}$, $\lambda_{12} = 2185.7\,\text{nm}$.
- **Physique du signal :** Le SWIR ($B_{11}$) est fortement absorbé par l'eau contenue dans les aiguilles.
- **Canopée saine et hydratée :** forte absorption SWIR $\rightarrow$ faible valeur de CRSWIR ($\sim 0.70 - 0.80$).
- **Dépérissement / Dessèchement :** perte d'eau dans les aiguilles $\rightarrow$ réflexion SWIR augmente $\rightarrow$ **le CRSWIR augmente significativement**.

---

## 2. Étape 3 : Régression Harmonique (Calibrage 2016)

L'objectif de l'étape 3 est d'apprendre sur une année de référence saine (2016) le **comportement phénologique normal** de chaque pixel afin de disposer d'une base de comparaison pour 2018.

### A. Fonction `compute_harmonic_terms(t, T=365.25)`
Pour un jour donné $t$ (nombre de jours depuis le `2015-01-01`), la fonction construit le vecteur de design $\mathbf{x}(t) \in \mathbb{R}^5$ :
$$\mathbf{x}(t) = \begin{bmatrix} 1 \\ \cos(\omega t) \\ \sin(\omega t) \\ \cos(2\omega t) \\ \sin(2\omega t) \end{bmatrix} \quad \text{avec } \omega = \frac{2\pi}{T} \text{ et } T = 365.25 \text{ jours}$$

* **Terme 1 (Intercept) :** représente le niveau moyen annuel de l'indice.
* **Termes 2 et 3 ($\cos(\omega t), \sin(\omega t)$) :** modélisent le cycle annuel principal (période de 1 an). La combinaison cosinus/sinus permet d'exprimer la courbe sous la forme $A_1 \cos(\omega t - \varphi_1)$, où $A_1 = \sqrt{a_1^2 + b_1^2}$ est **l'amplitude saisonnière** et $\varphi_1 = \arctan(b_1/a_1)$ est la **phase** (date du pic).
* **Termes 4 et 5 ($\cos(2\omega t), \sin(2\omega t)$) :** cycle semi-annuel (période de 6 mois), indispensable pour corriger les asymétries entre le reverdissement rapide de printemps et la sénescence progressive d'automne.

---

### B. Fonction `fit_harmonic_model(ts_days, ts_values, ts_masks)`
La fonction résout les moindres carrés ordinaires (OLS) **pixel par pixel** :

1. **Gestion des données manquantes et nuages :**
   Pour chaque pixel $j$, on extrait uniquement les dates valides (hors masques et non-NaN) : $n_{\text{valid}}$. On exige $n_{\text{valid}} > 5$ pour assurer l'identifiabilité du modèle à 5 paramètres.
2. **Matrice de Gram et estimation des coefficients $\hat{\beta}$ :**
   $$X_{\text{valid}} \in \mathbb{R}^{n_{\text{valid}} \times 5}$$
   $$\text{Gram} = X_{\text{valid}}^T X_{\text{valid}}$$
   $$\hat{\beta} = (\text{Gram})^{-1} X_{\text{valid}}^T \mathbf{y}_{\text{valid}}$$
3. **Variance des résidus et degrés de liberté :**
   $$df = n_{\text{valid}} - 5$$
   $$\hat{\sigma} = \sqrt{\frac{\sum_{i=1}^{n_{\text{valid}}} (y_i - \hat{y}_i)^2}{df}}$$

---

### C. Fonction `pred_harmonic_model(ts_days, fit_results)`
Pour n'importe quelle date future $t$ (notamment en 2018), la fonction calcule la prédiction et son **intervalle de prédiction à 95%** (fondé sur une loi de Student $t_{df}$) :

$$\hat{y}(t) = \mathbf{x}(t)^T \hat{\beta}$$
$$\Delta y(t) = t_{1 - \alpha/2, \, df} \cdot \hat{\sigma} \sqrt{1 + \mathbf{x}(t)^T (X^T X)^{-1} \mathbf{x}(t)}$$

* **D'où vient le terme $\sqrt{1 + \mathbf{x}^T (X^TX)^{-1} \mathbf{x}}$ ?**
  - Le terme $+1$ représente la **variabilité intrinsèque d'une nouvelle mesure** (variance $\sigma^2$).
  - Le terme $\mathbf{x}^T (X^TX)^{-1} \mathbf{x}$ représente **l'incertitude sur l'estimation des coefficients $\hat{\beta}$**.
  - Si l'on prédisait la moyenne, on n'aurait que le second terme (intervalle de confiance). Pour prédire une observation individuelle, on ajoute $+1$ (intervalle de prédiction).

---

## 3. Étape 5 : Méthode de Détection du Dépérissement (2018)

L'étape 5 est implémentée dans `anomaly_detection.py` (`detect_dieback`).

### Principe
1. On projette le modèle harmonique ajusté en 2016 sur l'ensemble des dates d'acquisition de 2018 :
   $$\text{Seuil Haut}(t) = \hat{y}_{2018}(t) + \Delta y_{2018}(t)$$
2. Un pixel est déclaré en **anomalie de dépérissement** à la date $t$ lorsque son humidité foliaire chute anormalement :
   $$\text{CRSWIR}_{\text{obs}}(t) > \text{Seuil Haut}(t)$$

### Sorties produites
* `dieback_map` (carte binaire) : $True$ si le pixel a connu au moins une anomalie significative au cours de l'année 2018.
* `anomaly_count` (carte de persistance) : nombre de dates où l'anomalie est confirmée. Permet d'éliminer les faux positifs isolés (ex. artefact de nuage résiduel).
* `max_anomaly` (carte de sévérité) : dépassement maximal observé au-dessus du seuil de tolérance.

---

## 4. Étape 6 : Discrimination Coupes Rases vs Scolytes

### La problématique
L'étape 5 détecte une perte de biomasse / d'humidité. Or, dans une forêt exploitée, **deux phénomènes produisent ce signal** :
1. Une **coupe rase** sanitaire ou d'exploitation (abattage total des arbres).
2. Une **attaque de scolytes** (arbres morts restant sur pied).

### Tableau comparatif des critères

| Critère | Coupe Rase (*Clearcut*) 🚜 | Attaque de Scolytes (*Bark Beetles*) 🪲 |
| :--- | :--- | :--- |
| **NDVI post-détection** | **Effondrement brutal : $NDVI < 0.30$** (sol minéral mis à nu). | **Maintien partiel : $NDVI > 0.35 - 0.40$** (arbres morts toujours sur pied, aiguilles rousses puis branches). |
| **Cinétique temporelle** | **Changement instantané** d'une image Sentinel-2 à la suivante (en 5 jours). | **Dégradation progressive** sur 4 à 8 semaines (baisse d'eau SWIR, puis rougissement visible, puis chute). |
| **Morphologie spatiale** | **Contours géométriques nets** (parcelles rectangulaires, linéaires de pistes forestières). | **Formes en "taches d'huile"** irrégulières, agrégées en foyers circulaires qui grossissent. |

### Implémentation algorithmique
```python
# Pseudo-code pour l'Étape 6
for pixel in pixels_anomalie_etape5:
    # 1. Vérification du NDVI post-anomalie
    if ndvi_2018[date_anomalie:, pixel].min() < 0.30:
        classification[pixel] = "Coupe rase"
    else:
        # L'arbre est toujours physiquement présent
        classification[pixel] = "Dépérissement sur pied (Candidat Scolyte)"
```

---

## 5. Étape 7 : Discrimination Sécheresse vs Scolytes

### La problématique
L'été 2018 a été marqué par une sécheresse et une canicule historiques. Tous les arbres ont subi un stress hydrique (augmentation généralisée du CRSWIR). Comment séparer un arbre qui a juste soif d'un arbre condamné par les scolytes ?

### Tableau comparatif des critères

| Critère | Sécheresse (*Drought*) ☀️ | Attaque de Scolytes (*Bark Beetles*) 🪲 |
| :--- | :--- | :--- |
| **Comportement en Automne** | **Réversible** : aux premières pluies de septembre/octobre, les arbres se réhydratent $\rightarrow$ le CRSWIR **redescend dans l'intervalle normal**. | **Irréversible** : l'arbre est mort (faisceaux conducteurs sectionnés) $\rightarrow$ le CRSWIR **reste définitivement en dehors du seuil**. |
| **Essence forestière touchée** | Touche indifféremment feuillus et résineux. | Spécifique aux **résineux / épicéas** (*Ips typographus* n'attaque pas les feuillus). |
| **Empreinte spatiale** | **Diffuse et régionale** : tout un versant ou massif réagit simultanément au déficit hydrique. | **Localisée en clusters** : foyers concentrés de mortalité au milieu d'arbres survivants. |

### Implémentation algorithmique

1. **Critère de réversibilité automnale :**
```python
# On analyse les dates d'octobre - novembre 2018
anomalie_automne = (obs_crswir[automne] > upper_threshold[automne])

if np.all(~anomalie_automne):
    classification[pixel] = "Stress sécheresse passager (Récupération)"
else:
    classification[pixel] = "Mortalité irréversible (Scolytes)"
```

2. **Filtre d'essence par l'amplitude 2016 de l'étape 3 :**
* Grâce à notre régression harmonique 2016, nous avons calculé l'amplitude annuelle :
  $$A = \sqrt{a_1^2 + b_1^2}$$
* **Feuillus (Hêtre, Chêne) :** forte amplitude saisonnière (perte complète des feuilles en hiver).
* **Conifères (Épicéas) :** faible amplitude saisonnière (aiguilles pérennes).
* **Règle :** Si un pixel en dépérissement présente une forte amplitude $A > A_{\text{seuil}}$, c'est un feuillu ayant souffert de la sécheresse, **pas une attaque de scolytes**.

---

## 6. Fiche Mémo / Discours pour la Présentation Orale

Voici la trame idéale pour expliquer le projet à votre professeur en 3 minutes de manière très claire :

> **1. Introduction & Contexte Local :**  
> *"Notre zone d'étude se situe à environ 30 km au sud d'Épinal, au cœur du massif vosgien. Ce secteur forestier à dominante d'épicéas a été le foyer d'une crise écologique et économique sans précédent en 2018 avec l'épidémie de scolytes. L'objectif est de mettre en place une chaîne de détection précoce automatisée par télédétection Sentinel-2, basée sur les principes de la méthodologie nationale FORDEAD."*

> **2. Modélisation Harmonique (Étape 3) :**  
> *"Sur l'année saine 2016, nous avons ajusté une régression harmonique à 5 termes par moindres carrés ordinaires sur l'indice CRSWIR (Continuum Removal SWIR). Le CRSWIR est idéal car il mesure directement la teneur en eau des aiguilles. Le modèle calcule pour chaque pixel le cycle saisonnier moyen et un intervalle de prédiction à 95% basé sur une loi de Student."*

> **3. Détection 2018 (Étape 5) :**  
> *"En projetant ce modèle sur 2018, toute sortie de l'intervalle de prédiction par le haut ($CRSWIR > \hat{y} + \text{seuil}$) traduit un dessèchement anormal de la canopée. Nous générons ainsi une carte de dépérissement et une carte de persistance de l'anomalie."*

> **4. Améliorations (Étapes 6 & 7) :**  
> *"Pour l'étape 6, nous séparons les coupes rases des scolytes grâce au NDVI : une coupe rase met le sol à nu ($NDVI < 0.3$), alors que le scolyte laisse l'arbre mort sur pied ($NDVI > 0.35$).  
> Pour l'étape 7, nous isolons la sécheresse des scolytes grâce à deux mécanismes : le test de réversibilité à l'automne (les arbres vivants récupèrent lors des pluies, les arbres scolytés restent morts) et le filtre d'essence via l'amplitude harmonique 2016, qui exclut les feuillus pour ne cibler que les épicéas."*

---
*Fichier généré automatiquement pour le projet Sentinel-2 — Novembre 2026.*
