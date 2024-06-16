import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_model(X, y):
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred)
    print(f'ROC AUC: {roc_auc}')

    return model

if __name__ == "__main__":
   
    X_train = pd.read_csv('data/X_train.csv')
    train_labels = pd.read_csv('data/training_set_labels.csv')

    
    model_xyz = train_model(X_train, train_labels['xyz_vaccine'])
    model_seasonal = train_model(X_train, train_labels['seasonal_vaccine'])

    
    import joblib
    joblib.dump(model_xyz, 'src/model_xyz.pkl')
    joblib.dump(model_seasonal, 'src/model_seasonal.pkl')
