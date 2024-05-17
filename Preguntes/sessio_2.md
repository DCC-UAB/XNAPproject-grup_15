# Preguntes fetes per la sessió 2:

## 1. Creació d'una Loss que utilitzi un mètode de classifcació però que pnalitzi les penalitzacions incorrectes :

**Objectiu:** La loss dels models actuals es calcula a partir de la cross entropy que és un metode de classificació. Estaria bé penalitzar les prediccions que s'allunyin molt de lobjectiu.

## 2. Datasets amb edats spearades:

**Objectiu:** Funciona igual de bé el model amb edats separedes (15,20,25,...) que amb edat juntes (15,16,17,18).

## 3. Biaixos Potencials:

**Objectiu:** Identificar possibles biaixos en els conjunts de dades que podrien afectar la precisió i l'objectivitat del model.

· Hi ha algun biaix evident en les dades (per exemple, més dades d'un grup d'edat específic, predominança d'un gènere)?

· Com podrien aquests biaixos afectar el rendiment del model a l'hora de predir l'edat en dades noves?

## 4. Soroll en les Imatges:

**Objectiu:** Detectar fonts de soroll que puguin afectar el rendiment del model i portar a sobreajustament o infraajustament.

· Les imatges presenten soroll visual, com ara marques d'aigua, ombres, il·luminació inconsistent o fons desordenats?

· En cas que hi hagi soroll, com es podria afectar el rendiment del model? Podria conduir a sobreajustament si el model aprèn a detectar elements irrellevants? O podria conduir a underfitting si el soroll dificulta l'aprenentatge de les característiques clau?
