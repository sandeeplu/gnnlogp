import pandas as pd



# Read the original ALIGNN dataset (no header, just id and property)

df = pd.read_csv("data.csv", header=None)



k = 3  # number of folds

for i in range(k):

    # Shuffle the rows deterministically for reproducibility

    df_fold = df.sample(frac=1, random_state=i).reset_index(drop=True)

    

    # Save each shuffled version as a new CSV (no header, no index)

    df_fold.to_csv(f"data_fold{i}.csv", index=False, header=False)


