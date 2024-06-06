# NOTES

4 Corves  (Evolució) 
- Metric Train
- Metric Test
- Loss Train 
- Loss Test

COSES A FER:
- 1. Obtenir Datasets Retallats
    - rang edats
    - aleatori --> CUIDADO!!! la distribucio OJO
    - edats concretes

- 2. Provar altres funcions COST (LOSS)
    - CE i CE modificats (els del START POINT)
    - MSE, MAE (per que el model consideri la diferència de edat de les predicts)

- 3. Provar diferents mètriques
    - L1/L2...

- 2. Canviar arquitectura model
    - Squezee
    - Canviar Resnet
    - canviar el model en si...

## Sessio 3

Noves tasques:
- 1. Dataset original vs Dataset Split propi (tot congelat)
- 2. Mirar arquitectures millor (tot congelat)
- 3. Afegir capes intermitjes a FC (Batchnorm, dropout...)
- 4. Prentrenar amb el model prentrenat (primer tot congelat per pretrain de mes descongelar) = Domain Adaptation
- 5. Provar DA al dataloader
- 6. LR policies (cos)
 
## Sessio 4
!!!Augmentar batch
Canviar a dataset CACD:
- 1. provar millors configuracions AFAD en CACD (primer amb model amb pretrain default)
- 2. provar diferents pretrains:
    - amb preentrenar amb model del 1.
    - preentrenar amb model tot congelat, descongelant 1 bloc +
    - preentrenar amb AFAD best model
- 3. provar predicts
      - mirar error per edats
      - mirar error per sexe
      - mirar casos especials (negre?)
- 4. extra:
        - fer detector cares
        - creuar train test dels datasets
        - ficar fotos de profes (obv la del Ramon això no es extra es obligatori xd)
    
## INDEX (versio final)
1- Mètriques i Loss 
2- Datasets
3- Models AFAD
4- Models CACD
5-Generalització dels models
6- Avaluació dels models
7- Prediccions extras
8- Conclusions

