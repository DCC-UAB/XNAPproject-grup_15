import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos de los archivos CSV
train_df = pd.read_csv('C:/Users/eduar/OneDrive/Escritorio/datasets/afad-propi-train.csv')
test_df = pd.read_csv('C:/Users/eduar/OneDrive/Escritorio/datasets/afad-propi-test.csv')

# Calcular la cantidad total de imágenes en cada conjunto de datos
total_images_train = len(train_df)
total_images_test = len(test_df)

# Calcular la cantidad de imágenes por cada edad en el conjunto de entrenamiento y prueba
train_image_counts = train_df['age'].value_counts()
test_image_counts = test_df['age'].value_counts()

# Calcular la probabilidad de cada edad en el conjunto de entrenamiento y prueba
train_age_probabilities = train_image_counts / total_images_train 
test_age_probabilities = test_image_counts / total_images_test 

# Normalizar las probabilidades para que sumen 1
train_age_probabilities_normalized = train_age_probabilities / train_age_probabilities.sum()
test_age_probabilities_normalized = test_age_probabilities / test_age_probabilities.sum()

# Lista de edades reales para el conjunto de entrenamiento (edades originales + 15)
edades_train = train_age_probabilities_normalized.index + 15

# Lista de edades reales para el conjunto de prueba (edades originales + 15)
edades_test = test_age_probabilities_normalized.index + 15

# Ancho de las barras
bar_width = 0.35

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))

# Graficar las barras para el conjunto de entrenamiento
plt.bar(edades_train - bar_width/2, train_age_probabilities_normalized.values, label='Entrenament', color='red', alpha=0.7, width=bar_width)

# Graficar las barras para el conjunto de prueba
plt.bar(edades_test + bar_width/2, test_age_probabilities_normalized.values, label='Prova', color='blue', alpha=0.7, width=bar_width)

# Configurar etiquetas y título
plt.title("Distribució d'edats al dataset propi")
plt.xlabel('Edat')
plt.ylabel('Percentatge sobre 1')
plt.xticks(edades_train)
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()



# Lista de valores para el segundo gráfico (el que ya estaba)
test = [9.407, 8.483, 7.43, 6.41, 5.711, 4.775, 3.94, 3.204, 2.589, 1.942, 
        1.622, 1.596, 1.807, 2.313, 2.855, 3.665, 4.6, 5.412, 6.381, 7.438, 
        8.38, 9.436, 10.26, 11.217, 12.366, 12.746]
train = [9.282, 8.382, 7.462, 6.353, 5.665, 4.744, 3.862, 3.16, 2.501, 1.938,
         1.574, 1.529, 1.853, 2.304, 2.997, 3.906, 4.757, 5.625, 6.646, 7.622,
         8.547, 9.504, 10.378, 11.377, 12.365, 13.126]

# Números del 15 al 40
numeros = list(range(15, 41))

# Crear el segundo gráfico de barras con los valores dados
plt.figure(figsize=(12, 8))
bar_width = 0.4
indices = range(len(test))

# Graficar las barras
plt.bar(indices, test, width=bar_width, label='Test', color='blue')
plt.bar([i + bar_width for i in indices], train, width=bar_width, label='Train', color='orange')

# Configurar etiquetas y título
plt.title('MAE per edats epoch 45')
plt.xlabel('Edats')
plt.ylabel('MAE')
plt.xticks([i + bar_width / 2 for i in indices], numeros)
plt.legend()

# Mostrar el gráfico
plt.show()
