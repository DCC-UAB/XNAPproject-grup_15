# AGE PREDICTOR USING REGRESSION MODEL

**Autors:** Eduard Ara (1636341), Abril Batalla (1638650), Francesca Company (1598017), Andreu Mir (1637558)  
**Grup:** 15  
**Curs:** Xarxes Neuronals 2023-2024  
**Professors:** Ramon Baldrich, Carlos Boned  

## Introducció

Aquest projecte es basa en el desenvolupament i anàlisi d'un model de xarxes neuronals de regressió per a la predicció de l'edat de les persones a partir de les seves cares. Des d'un principi, vam establir com a objectiu obtenir el millor model de regressió possible, provant diferents configuracions dels paràmetres i observant quins canvis proporcionaven un millor rendiment. Això ens va permetre analitzar els resultats i extreure conclusions sobre el funcionament del nostre model.

### 1.2 Observacions del Punt de Partida

Inicialment, vam començar amb un model classificador que utilitzava Cross Entropy com a funció de pèrdua. Aquest va ser el primer aspecte que vam haver de tractar al projecte, ja que ens va sorprendre que Cross Entropy pogués funcionar bé.

La raó principal d'aquesta incertesa és que Cross Entropy penalitza de la mateixa manera un error d'edat de 2 anys que un de 40 anys, ja que, sent un classificador, es centra en classificar en categories i no valora la distància entre aquestes.

Per resoldre aquest problema, el mateix model de partida està implementat amb dues variants de Cross Entropy que penalitzen la diferència d'edat. Aquestes funcionen mitjançant la creació d'un tensor binari ordinal. Aquest tensor, de longitud igual al nombre d'edats possibles, tenia 1's fins a arribar a l'edat que es vol representar, i la resta amb 0 (per exemple, l'edat de 5 anys seria [1,1,1,1,1,1,0,0,0...]). Vam trobar diversos problemes de compatibilitat amb alguns mòduls del codi, pel que no vam poder executar correctament aquestes variants. No obstant això, vam observar els resultats que havien obtingut altres usuaris, i no eren dolents.

Malgrat tot, vam decidir continuar amb la idea de provar un model regressor, ja que vam veure que el model classificador ja havia estat explorat exhaustivament, i hi havia molts projectes i estudis sobre el seu funcionament en totes les seves variants.

### 1.3 Mètriques i Funció de Pèrdua

Per tant, vam escollir com a funció de pèrdua la més utilitzada en models regressors d'aquest tipus, l'Error Quadràtic Mitjà (MSE). Com a mètrica, vam seleccionar l'Error Absolut Mitjà (MAE), ja que permet veure exactament l'error d'edat en anys que comet el model. També vam utilitzar MSE com a mètrica, però només ens hi fixàvem en casos d'overfitting, ja que permet veure de manera més clara si aquest existeix i en quin grau.

També vam provar com a mètrica el coeficient de determinació R^2, ja que inicialment vam provar un conjunt de dades reduït de manera aleatòria, i inicialment pensàvem que aquesta mètrica ens ajudaria a valorar també la variància de les dades. Després d'implementar-la, però, vam veure que no era tan bona idea com pensàvem i la vam eliminar.

## Conjunts de Dades

Per a aquest projecte vam treballar amb dos conjunts de dades. Són dels dos conjunts de dades públiques més grans que existeixen que contenen imatges de cares.

### 2.1 AFAD (Asian Face Age Detection)

Aquest conjunt de dades conté aproximadament 160.000 imatges amb edats dels 15 als 70 anys de cares de persones asiàtiques. És un conjunt de dades molt complet, que inclou camps per a l'edat (com a etiquetes que comencen en 0) i el gènere de la persona de la imatge.

El problema amb aquest conjunt de dades és la distribució de les dades, ja que gairebé el 40% de les dades es concentren en la franja dels 20 als 30 anys. Aquest fet ens va fer optar per retallar el rang d'edats utilitzades per a l'entrenament, prenent les edats de 15 a 40. A continuació mostrarem les gràfiques de la distribució de les dades.

Com podem observar, la majoria de dades es concentren en el rang dels 10 als 26. A més, veiem com el conjunt de validació només té 4 edats, pel que era completament inútil. Per això vam optar per fer una nova distribució del conjunt de dades, unint totes les dades que teníem (des del rang de 15 a 40), i fent un nou split, únicament entre entrenament i proves, del 70/30. A més, vam fer una prova inicial del rendiment amb aquest nou conjunt de dades i amb l'original, utilitzant un model ResNet101 amb totes les capes congelades excepte la fully-connected.

Com podem observar, tots dos tenen un rendiment similar, ja que simplement hem redistribuït les dades. Vam optar pel conjunt de dades propi, per qüestions de practicitat, ja que al no utilitzar el conjunt de validació, tindríem les dades d'aquest per als altres conjunts.

### 2.2 CACD (Cross Age Celebrity Detection)

Aquest és el segon conjunt de dades que tenim, amb imatges de 160.000 celebritats, des dels 15 fins als 77 anys. El principal problema amb aquest conjunt de dades és que les imatges estan fetes totes per professionals, i per a professionals. És a dir, que són molt diferents a les imatges casuals que algú podria fer a casa, limitant així les possibles aplicacions d'un model amb aquest conjunt de dades.

En aquest cas, tornem a tenir una mala distribució de les edats, però no tan exagerada com en el cas de l'AFAD. Igualment vam optar per retallar una mica el conjunt de dades, i quedar-nos amb una distribució prou equilibrada, que comprenia edats dels 15 als 68. Cal mencionar que aquest conjunt de dades no inclou el camp de gènere. A continuació es mostra la distribució de les dades per les edats seleccionades pel nostre model. Es pot veure com està molt més equilibrat que l'AFAD.

[CONTINUACIÓ AL PDF](**https://github.com/DCC-UAB/XNAPproject-grup_15/blob/47ac573cdcbd27594ba41bd8dc154e7d203adbe4/Age_Prediction-Grup15.pdf**)

