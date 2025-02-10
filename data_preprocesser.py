import sys
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python3 data_scaler.py [raw data (.csv)] [scaling method: either 'standard' or 'minmax']\n")
        sys.exit(1)

    elif sys.argv[2] not in ['standard', 'minmax']:
        sys.stderr.write("Usage: python3 data_scaler.py [raw data (.csv)] [scaling method: either 'standard' or 'minmax']\n")
        sys.exit(1)

    else:
        # Read the data
        input_csv = sys.argv[1]
        df = pd.read_csv(input_csv)

        # Split features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        scale_method = sys.argv[2]

        # Perform scaling
        if scale_method == 'standard':
            scaler = StandardScaler()
        
        elif scale_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler.fit_transform(X)

        # Combine scaled features and target back into a dataframe
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        df_scaled['y'] = y

        # Output
        df_scaled.to_csv(input_csv, index=False)
