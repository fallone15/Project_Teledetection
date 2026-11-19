1. Download Sentinel-2 images from 2016/01/01 to 2017/01/01.
D'abord nous avons ajouté un path dans le fichier yaml dans lequel se telechargera le dossier SENTINEL L2A ensuite avons executé 
2. Preprocess the data by masking clouds, and non-vegetation pixels.
Nous avons converti les valeurs des bandes Sentinel‑2 en réflectance, puis calculé le NDVI (`B08`, `B04`). Pour chaque image, nous combinons les critères suivants :
- **Nuages** : nous repérons les pixels très lumineux dans les bandes visibles `B02`, `B03` et `B04` (`> 0.2`, brillance `> 0.25`), avec un NDVI faible (`< 0.4`), et nous confirmons la présence de nuages avec la bande SWIR (`B11 > 0.15`).
- **Sols et autres zones non végétalisées** : nous utilisons un seuil de NDVI de `0.3` complété par la bande SWIR (`B11 > 0.20`), car les sols nus et surfaces minérales présentent une forte réflectance dans le SWIR contrairement à la végétation.
Les pixels exclus et les données manquantes sont mis à `NaN` dans la série de NDVI afin de les ignorer lors de la régression. Nous avons vérifié le masque avec `verify.py` en comparant l'image RGB originale et la même image avec les pixels exclus en rouge.
3. Fit a regression model on the 2016 data.
4. Download Sentinel-2 images from 2018/01/01 to 2019/01/01.
5. Design a method to detect tree dieback, and apply on the 2018 data.*
Nous avons implémenté la détection dans `anomaly_detection.py` (`detect_dieback`). Pour chaque date de 2018, nous comparons le CRSWIR observé aux prédictions du modèle harmonique calibré sur 2016. Une anomalie de dépérissement est détectée lorsque le CRSWIR réel dépasse le seuil haut de l'intervalle de prédiction à 95% (`CRSWIR_obs > pred + interval`). La méthode produit une carte binaire des surfaces atteintes ainsi qu'une carte du nombre d'anomalies observées au cours de l'année 2018.
6. Improve the method to distinguish between clearcut harvesting and bark beetle infestations.
7. Improve the method to distinguish between droughts and bark beetle infestations.
