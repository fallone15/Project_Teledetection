1. Download Sentinel-2 images from 2016/01/01 to 2017/01/01.
D'abord nous avons ajouté un path dans le fichier yaml dans lequel se telechargera le dossier SENTINEL L2A ensuite avons executé 
2. Preprocess the data by masking clouds, and non-vegetation pixels.
Nous avons converti les valeurs des bandes Sentinel‑2 en réflectance, puis calculé le NDVI à partir de la bande rouge (`B04`) et du proche infrarouge (`B08`). Pour chaque image, nous avons combiné trois critères de masquage :
- **Nuages** : nous repérons les pixels très lumineux dans les bandes visibles `B02`, `B03` et `B04`, avec un NDVI faible.
- **Sols et autres zones non végétalisées** : nous utilisons un seuil de NDVI de `0,3` pour les écarter.
Les pixels exclus et les données manquantes sont mis à `NaN` dans la série de NDVI afin de les ignorer lors de la régression. Nous avons vérifié le masque en comparant, pour plusieurs dates, l’image RGB originale et la même image avec les pixels exclus en rouge. Cette comparaison montre que certains nuages restent visibles, notamment sur l’image du 14 mars 2017.
3. Fit a regression model on the 2016 data.
4. Download Sentinel-2 images from 2018/01/01 to 2019/01/01.
5. Design a method to detect tree dieback, and apply on the 2018 data.*
6. Improve the method to distinguish between clearcut harvesting and bark beetle infestations.
7. Improve the method to distinguish between droughts and bark beetle infestations.
