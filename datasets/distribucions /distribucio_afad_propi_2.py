import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos de los archivos CSV
train_df = pd.read_csv('C:/Users/eduar/OneDrive/Escritorio/datasets/afad-propi-train.csv')
test_df = pd.read_csv('C:/Users/eduar/OneDrive/Escritorio/datasets/afad-propi-test.csv')

# Calcular la cantidad total de imágenes en cada conjunto de datos
total_images_train = len(train_df)
total_images_test = len(test_df)

# Calcular la cantidad de imágenes por género en el conjunto de entrenamiento y prueba
train_gender_counts = train_df['gender'].value_counts()
test_gender_counts = test_df['gender'].value_counts()

# Calcular la probabilidad de cada género en el conjunto de entrenamiento y prueba
train_gender_probabilities = train_gender_counts / total_images_train 
test_gender_probabilities = test_gender_counts / total_images_test 

# Combinar las probabilidades de género de entrenamiento y prueba en un solo DataFrame
combined_df = pd.concat([train_gender_probabilities, test_gender_probabilities], axis=1)
combined_df.columns = ['Entrenamiento', 'Prueba']

# Graficar las probabilidades de género en un solo plot
combined_df.plot(kind='bar', figsize=(10, 6))
plt.title('Distribución de género en los conjuntos de entrenamiento y prueba')
plt.xlabel('Género')
plt.ylabel('Probabilidad')
plt.xticks(rotation=0)  # Para evitar que los nombres de los géneros estén inclinados
plt.legend(title='Conjunto de datos')
plt.show()
