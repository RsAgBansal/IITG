import pandas as pd
import joblib

def make_predictions():
    X_test = pd.read_csv('data/X_test.csv')
    test_features = pd.read_csv('data/test_set_features.csv')

    # Load models
    model_xyz = joblib.load('src/model_xyz.pkl')
    model_seasonal = joblib.load('src/model_seasonal.pkl')

    # Make predictions
    test_predictions_xyz = model_xyz.predict_proba(X_test)[:, 1]
    test_predictions_seasonal = model_seasonal.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission = pd.DataFrame({
        'respondent_id': test_features['respondent_id'],
        'xyz_vaccine': test_predictions_xyz,
        'seasonal_vaccine': test_predictions_seasonal
    })

    submission.to_csv('submission/submission.csv', index=False)
    print(submission.head())

if __name__ == "__main__":
    make_predictions()
