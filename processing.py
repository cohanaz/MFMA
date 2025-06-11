import numpy as np
import pandas as pd

# Create a test set for a row of real data with guassian noise
def create_augmented_set(row, n_cases=3, feature_scale=0.2):
    augmented_set = [row]

    # Make copies of row
    for _ in range(n_cases):
        # Standardize the original row and use the calculated std for the normal distribution
        #scaler = StandardScaler().fit(np.array([row]))
        #standardized_row = scaler.transform(np.array([row]))[0]
        
        # Create vector of random gaussians using the standardized std
        noise = np.random.normal(loc=0.0, scale=feature_scale*np.abs(row), size=len(row))
        #noise = np.random.uniform(low=-feature_scale*np.abs(row), high=feature_scale*np.abs(row), size=len(row))
        
        # Add to the standardized row to get the new row
        new_row = row + noise
        
        # Store in test set
        augmented_set.append(new_row)

    return augmented_set

def process_row(args):
    index, row, feature_scale, target_model, n_cases = args
    augmented_set = create_augmented_set(list(row), n_cases=n_cases, feature_scale=feature_scale)
    augmented_set_df = pd.DataFrame(augmented_set, columns=row.index)
    preds = target_model.predict(augmented_set_df)
      
    orig_pred = preds[0]
    aug_preds = preds[1:]
    pred_range = np.max(aug_preds) - np.min(aug_preds)
    diff = abs(orig_pred - np.mean(aug_preds))

    # Flattened prediction vector
    pred_features = list(aug_preds)  # Now each prediction becomes its own return value

    # Return: scalar metrics + each prediction as a separate column
    return (index, np.var(aug_preds), pred_range, diff, *pred_features)

