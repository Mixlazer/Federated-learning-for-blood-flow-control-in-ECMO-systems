import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import uuid

def train_model(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    X = df[["age", "height", "weight"]]  # упрощенно
    y = df["rpm"]

    # Старт логирования
    with mlflow.start_run(run_name="retrain_" + str(uuid.uuid4())[:8]):
        model = Ridge()
        model.fit(X, y)

        # Предсказания и метрики
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        # Логируем
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained and logged. MSE: {mse}")
        return model
