import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(features):
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = features.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def load_data():
    train_features = pd.read_csv('data/training_set_features.csv')
    train_labels = pd.read_csv('data/training_set_labels.csv')
    test_features = pd.read_csv('data/test_set_features.csv')

    return train_features, train_labels, test_features

if __name__ == "__main__":
    train_features, train_labels, test_features = load_data()
    preprocessor = preprocess_data(train_features.drop('respondent_id', axis=1))
    X_train = preprocessor.fit_transform(train_features.drop('respondent_id', axis=1))
    X_test = preprocessor.transform(test_features.drop('respondent_id', axis=1))

   
    pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/X_test.csv', index=False)
