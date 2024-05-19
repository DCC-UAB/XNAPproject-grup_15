# Preguntes fetes per la sessió 2:

## 1. Comparació dels resultats utilitzant el MSE com a Loss o el mètode propi com a Loss

**Objectiu:** Comparar els dos models per veure quin presenta millors resultats.

## 2. Datasets amb edats separades:

**Objectiu:** Funciona igual de bé el model amb edats separades (15, 20, 25, ...) que amb edats agrupades (15, 16, 17, 18)?

## 3. Overfitting:

**Objectiu:** El model presenta un sobreajustament molt clar, primer de tot haurem de reduir-lo.

· El starting point no utilitza una arquitectura preentrenada -> Això provoca una fàcil memorització de les imatges i per tant pot ser la causa del sobreajustament.

· El dataset original presenta una mala distribució en els conjunts d'entrenament, test i validació. Realitzar una nova distribució 70% 30% entre entrenament i test per intentar solucionar el sobreajustament.

· Si l'anterior no funciona es pot provar a realitzar un data augmentation, una extracció de característiques congelant blocs o altres mètodes per reduir el sobreajustament.

## 4. Noves mètriques

· R² -> Aquesta mètrica indica quina proporció de la variància en la variable dependent (en aquest cas, l'edat) és explicada pel model. Un R² més alt significa que el model explica millor la variància de les dades. Interessant per a datasets amb un ampli rang d'edats.



