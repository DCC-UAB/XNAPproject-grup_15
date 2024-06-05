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
plt.bar(edades_train - bar_width/2, train_age_probabilities_normalized.values, label='Conjunto de entrenamiento', color='skyblue', alpha=0.7, width=bar_width)

# Graficar las barras para el conjunto de prueba
plt.bar(edades_test + bar_width/2, test_age_probabilities_normalized.values, label='Conjunto de prueba', color='lightgreen', alpha=0.7, width=bar_width)

# Configurar etiquetas y título
plt.title("Distribució d'edats al dataset propi")
plt.xlabel('Edat')
plt.ylabel('Percentatge sobre 1')
plt.xticks(edades_train)
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()
