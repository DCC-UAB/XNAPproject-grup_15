# Preguntes fetes per la sessió 2:

## 1. Creació d'una Loss que utilitzi un mètode de classifcació però que pnalitzi les penalitzacions incorrectes :

**Objectiu:** La loss dels models actuals es calcula a partir de la cross entropy que és un metode de classificació. Estaria bé penalitzar les prediccions que s'allunyin molt de lobjectiu.

## 2. Datasets amb edats spearades:

**Objectiu:** Funciona igual de bé el model amb edats separedes (15,20,25,...) que amb edat juntes (15,16,17,18).

## 3. Altres mètriques:

**Objectiu:** Utilitzem MAE i MSE com a mètriques per veure el rendiment però estaria bé afegir d'altres .

· R2 -> Aquesta mètrica indica quina proporció de la variància en la variable dependent (en aquest cas, l'edat) és explicada pel model. Un R² més alt significa que el model explica millor la variància de les dades. Interesant per datasets amb ampli rang d'edats

· MAPE -> És robust a valors atípics i proporciona la mediana de l'error absolut. Això significa que és més resistent als valors extrems comparat amb altres mètriques.


