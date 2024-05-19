# Preguntes fetes per la sessió 1:

## 1. Provar diferents arquitectures:

**Objectiu:** Provar diverses arquitectures per determinar quina ofereix els millors resultats en termes de rendiment i precisió.

· Realitzar un anàlisi exhaustiu dels resultats obtinguts de cadascuna de les arquitectures provades.

· Ajustar i modificar les arquitectures segons els resultats de l'anàlisi per millorar el rendiment i l'eficàcia.

## 2. Probar diferents datasets:

**Objectiu:** Per fer les proves inicialment hem utilitzat tot el dataset però el temps és massa elevat ja que ha de processar 165K imatges. Poder provar amb datasets més petits de diferents característiques.

· Provar amb un dataset que només presenti un rang d'edats contínues (per exemple, de 15 a 25).

· Provar amb un dataset que agafi aleatòriament el nombre d'exemples de cada edat, tenint en compte que segueixin una mateixa distribució.

## 3. Classificador com a Loss?

**Objectiu:** Els starting points utilitzen un classificador per calcular la loss. Això no concorda amb el nostre objectiu ja que només mira si s'ha classificat bé o no. Nosaltres volem aproximar-nos a l'edat el màxim possible, per tant, necessitem una regressió o un nou mètode per calcular la loss.

· Calcular la loss amb el MSE.

· Nou mètode per calcular la loss.






