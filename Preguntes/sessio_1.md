# Preguntes fetes per la sessió 1:

Durant aquesta primera sessió, ens volem centrar en analitzar les dades que tenim. Això ens ajudarà a definir el nostre enfocament per a les pròximes sessions i elaborar un model que s'adapti al nostre problema. En el nostre cas, disposem de tres fonts de dades diferents, cadascuna amb el seu propi conjunt d'entrenament, de test i de validació. Aquestes fonts de dades són AFAD, CACD i MORPH. 

**AFAD (Asian Face Age Dataset):** És un conjunt de dades que conté imatges facials d'individus asiàtics classificades per edat. Sovint s'utilitza per a tasques de predicció d'edat a partir d'imatges facials.

**CACD (Cross-Age Celebrity Dataset):** És un conjunt de dades que conté imatges facials de celebritats en diferents edats. És útil per a tasques de reconeixement facial i predicció d'edat, especialment per la seva diversitat i variabilitat temporal.

**MORPH:** És un conjunt de dades que també conté imatges facials de persones en diverses etapes de la seva vida. Sol usar-se per a estudis de predicció d'edat i anàlisi del canvi facial al llarg del temps.

A partir d'aquestes fonts, volem respondre les següents preguntes:


## 1. Distribució d'Edats:

**Objectiu:** Assegurar que el model tingui prou dades de cada rang d'edat per aprendre adequadament i no esbiaixar-se cap a certs grups.

· Quin és el rang d'edats en cada conjunt (train, test i validation) per a cadascuna de les fonts de dades (AFAD, CACD, MORPH)?
· Està equilibrada la distribució d'edats en cada conjunt? Si no, hi ha grups d'edat amb una major representació que d'altres?
- comparar distribucions d'edats entre groundtruth i predicted

## 2. Quantitat de Dades:

**Objectiu:** Comprovar que hi ha prou dades per entrenar, validar i testejar el model, evitant el sobreajustament i l'infraajustament.

· Quin és el nombre d'imatges en cada conjunt (train, test, validation) per a cada font de dades? (10%,10%,80%)

· El conjunt d'entrenament (train) és prou gran per evitar el sobreajustament, és a dir, que el model no aprengui només les característiques específiques de les dades d'entrenament?

· El conjunt de validació i test tenen suficients dades per detectar el sobreajustament i evitar que el model tingui un rendiment baix en dades noves (underfitting)?

## 3. Biaixos Potencials:

**Objectiu:** Identificar possibles biaixos en els conjunts de dades que podrien afectar la precisió i l'objectivitat del model.

· Hi ha algun biaix evident en les dades (per exemple, més dades d'un grup d'edat específic, predominança d'un gènere)?

· Com podrien aquests biaixos afectar el rendiment del model a l'hora de predir l'edat en dades noves?

## 4. Soroll en les Imatges:

**Objectiu:** Detectar fonts de soroll que puguin afectar el rendiment del model i portar a sobreajustament o infraajustament.

· Les imatges presenten soroll visual, com ara marques d'aigua, ombres, il·luminació inconsistent o fons desordenats?

· En cas que hi hagi soroll, com es podria afectar el rendiment del model? Podria conduir a sobreajustament si el model aprèn a detectar elements irrellevants? O podria conduir a underfitting si el soroll dificulta l'aprenentatge de les característiques clau?



- que passa quan fas data augmentation, feature extraction, fine tuning?
- ficar arquitectures diferents a la resnet34 , per exemple resnet 50










