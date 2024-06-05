import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos de los archivos CSV
train_df = pd.read_csv('afad_train.csv')
test_df = pd.read_csv('afad_test.csv')
valid_df = pd.read_csv('afad_valid.csv')

# Concatenar los conjuntos de datos para obtener todas las edades presentes
all_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

# Calcular la cantidad total de imágenes en cada conjunto de datos
total_images_train = len(train_df)
total_images_test = len(test_df)
total_images_valid = len(valid_df)
total_images_all = len(all_df)

# Calcular la cantidad de imágenes por cada edad en el conjunto de entrenamiento, prueba y validación
train_image_counts = train_df['age'].value_counts()
test_image_counts = test_df['age'].value_counts()
valid_image_counts = valid_df['age'].value_counts()

# Obtener todas las edades presentes en los conjuntos de entrenamiento, prueba y validación
all_ages = sorted(set(train_image_counts.index) | set(test_image_counts.index) | set(valid_image_counts.index))

# Llenar con 0 las edades faltantes en el conjunto de entrenamiento
train_age_probabilities_normalized = train_image_counts.reindex(all_ages, fill_value=0) / total_images_train
test_age_probabilities_normalized = test_image_counts.reindex(all_ages, fill_value=0) / total_images_test
valid_age_probabilities_normalized = valid_image_counts.reindex(all_ages, fill_value=0) / total_images_valid

# Sumar 15 a todas las edades para tener las edades reales
all_ages = [age + 15 for age in all_ages]

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))

# Graficar las barras para el conjunto de entrenamiento
plt.bar([age - 0.2 for age in all_ages], train_age_probabilities_normalized, width=0.2, color='red', label='Entrenament')

# Graficar las barras para el conjunto de prueba
plt.bar(all_ages, test_age_probabilities_normalized, width=0.2, color='blue', label='Prova')

# Graficar las barras para el conjunto de validación
plt.bar([age + 0.2 for age in all_ages], valid_age_probabilities_normalized, width=0.2, color='orange', label='Validació')

# Configurar etiquetas y título
plt.title("Distribució d'edats al dataset original")
plt.xlabel('Edat')
plt.ylabel('Probabilitat')
plt.xticks(all_ages)

# Añadir la leyenda
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()
