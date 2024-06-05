import matplotlib.pyplot as plt
import pandas as pd

# Llegir les dades dels arxius CSV
train_df = pd.read_csv('cacd_train.csv')
test_df = pd.read_csv('cacd_test.csv')
valid_df = pd.read_csv('cacd_valid.csv')

# Calcular la quantitat total d'imatges en cada conjunt de dades
total_images_train = len(train_df)
total_images_test = len(test_df)
total_images_valid = len(valid_df)

# Calcular la quantitat d'imatges per cada edat en el conjunt d'entrenament, prova i validació
train_image_counts = train_df['age'].value_counts()
test_image_counts = test_df['age'].value_counts()
valid_image_counts = valid_df['age'].value_counts()

# Calcular la probabilitat de cada edat en el conjunt d'entrenament, prova i validació
train_age_probabilities = train_image_counts / total_images_train 
test_age_probabilities = test_image_counts / total_images_test 
valid_age_probabilities = valid_image_counts / total_images_valid

# Obtenir totes les edats presents en qualsevol dels conjunts
all_ages = sorted(set(train_age_probabilities.index).union(set(test_age_probabilities.index)).union(set(valid_age_probabilities.index)))

# Crear un DataFrame amb les probabilitats normalitzades per edat per a cada conjunt
age_data = pd.DataFrame(index=all_ages)
age_data['train'] = train_age_probabilities
age_data['test'] = test_age_probabilities
age_data['valid'] = valid_age_probabilities

# Omplir els valors NaN amb 0
age_data = age_data.fillna(0)

# Normalitzar les probabilitats per tal que sumin 1
train_age_probabilities_normalized = age_data['train'] / age_data['train'].sum()
test_age_probabilities_normalized = age_data['test'] / age_data['test'].sum()
valid_age_probabilities_normalized = age_data['valid'] / age_data['valid'].sum()

# Amplada de les barres
bar_width = 0.25

# Posicions de les barres
r1 = range(len(all_ages))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Crear el gràfic de barres
plt.figure(figsize=(12, 6))

# Graficar les barres per al conjunt d'entrenament
plt.bar(r1, train_age_probabilities_normalized.values, label='Conjunt d\'entrenament', color='red', alpha=0.7, width=bar_width)

# Graficar les barres per al conjunt de prova
plt.bar(r2, test_age_probabilities_normalized.values, label='Conjunt de prova', color='blue', alpha=0.7, width=bar_width)

# Graficar les barres per al conjunt de validació
plt.bar(r3, valid_age_probabilities_normalized.values, label='Conjunt de validació', color='orange', alpha=0.7, width=bar_width)

# Configurar etiquetes i títol
plt.title("Distribució d'edats al dataset originals")
plt.xlabel('Edat')
plt.ylabel('Percentatge sobre 1')
plt.xticks([r + bar_width for r in range(len(all_ages))], [age + 15 for age in all_ages])
plt.legend()

# Mostrar el gràfic
plt.tight_layout()
plt.show()
