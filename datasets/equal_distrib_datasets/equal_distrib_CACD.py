import pandas as pd
import numpy as np
import random

adad_train_path = 'datasets/cacd-train1.csv'
adad_test_path = 'datasets/cacd-test1.csv'

# Read the data from the CSV files
train_df = pd.read_csv(adad_train_path)
test_df = pd.read_csv(adad_test_path)

def calculate_mean(df):
    # Calculate the number of rows per age
    rows_per_age = df['age'].value_counts()

    # Calculate the mean number of rows per age
    mean_rows_per_age = rows_per_age.mean()

    return mean_rows_per_age

for i,df in enumerate([train_df, test_df]):

    # Calculate the number of rows per age
    rows_per_age = df['age'].value_counts()
    ages_to_decrease = rows_per_age[rows_per_age > 1300].index

    # Calculate the number of rows to decrease per age
    rows_to_decrease = rows_per_age[ages_to_decrease] - 1150

    # Create a new DataFrame to store the adjusted dataset
    df_adjusted = pd.DataFrame(columns=df.columns)

    # Iterate over the ages to decrease
    for age in ages_to_decrease:
        # Get the rows with the current age
        age_rows = df[df['age'] == age]

        # Randomly select the rows to keep
        rows_to_keep = np.random.choice(age_rows.index, random.randint(1000,1300), replace=False)

        # Add the selected rows to the new DataFrame
        df_adjusted = pd.concat([df_adjusted, df.loc[rows_to_keep]])

    # Add the rows of the ages that do not need to be decreased
    other_ages = rows_per_age[rows_per_age <= 1300].index
    for age in other_ages:
        df_adjusted = pd.concat([df_adjusted, df[df['age'] == age]])

    # Guardar el nuevo dataset ajustado
    if i == 0:
        df_adjusted.to_csv("datasets/equal_distrib_datasets/cacd-train-equi.csv", index=False)
        print("Dataset TRAIN ajustado guardado.")
    else:
        df_adjusted.to_csv("datasets/equal_distrib_datasets/cacd-test-equi.csv", index=False)
        print("Dataset TEST ajustado guardado.")