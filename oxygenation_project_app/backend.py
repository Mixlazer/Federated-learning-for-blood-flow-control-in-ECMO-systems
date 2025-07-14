import os
import uuid
import numpy as np
import pandas as pd
import math
import pickle
from keras.models import load_model, save_model, Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import shutil


class OxygenBackend:
    def __init__(self):
        self.versions_file = "models/versions.csv"
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(self.versions_file):
            with open(self.versions_file, "w", encoding="utf-8", newline="\n") as f:
                f.write("version_id,created_at,mae,val_loss,train_loss,scaler_path\n")

        # Проверяем наличие базовой модели
        self.ensure_initial_model()

    def ensure_initial_model(self):
        """Создает базовую модель и scaler, если они отсутствуют"""
        initial_model_path = "models/initial_model.keras"
        initial_scaler_path = "models/initial_model.scaler"

        if not os.path.exists(initial_model_path):
            raise FileNotFoundError(f"initial model or scaler deleted")

    def calculate_rpm(self, height, weight, pump_type, model_name):
        pump_map = {"type1": 1, "type2": 2, "type3": 3}
        pump = pump_map.get(pump_type, 1)
        bsa = math.sqrt(height * weight / 3600)
        flow = bsa * 2.4
        X_new = np.array([[height, weight, bsa, flow, pump]])

        # Загружаем модель и соответствующий scaler
        model_path = os.path.join("models", model_name)
        scaler_path = self.get_scaler_path_for_model(model_name)

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found for model: {model_name}")

        with open(scaler_path, "rb") as f:
            scaler_data = pickle.load(f)
            x_scaler = scaler_data['x_scaler']
            y_scaler = scaler_data['y_scaler']

        # Масштабируем входные данные
        X_new_scaled = x_scaler.transform(X_new)

        # Загружаем модель и делаем предсказание
        model = load_model(model_path)
        rpm_scaled = model.predict(X_new_scaled)
        rpm = y_scaler.inverse_transform(rpm_scaled)[0][0]

        return rpm

    def get_scaler_path_for_model(self, model_name):
        """Возвращает путь к scaler для указанной модели"""
        base_name = os.path.splitext(model_name)[0]
        return os.path.join("models", f"{base_name}.scaler")

    def load_and_fedavg_model(self, file_path, validation_split, epochs, batch_size, early_stop_enabled):
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        required_columns = ['Age', 'Height(cm)', 'Weight(Kg)', 'BSA(m2)',
                            'Calculated Flow(Lit/min)', 'AVG RPM', 'Pump Type']
        if not all(col in df.columns for col in required_columns):
            raise Exception("Файл не содержит всех необходимых столбцов")

        df.columns = required_columns
        df = df.rename(columns={
            'Height(cm)': 'Height',
            'Weight(Kg)': 'Weight',
            'BSA(m2)': 'BSA',
            'Calculated Flow(Lit/min)': 'Flow',
            'AVG RPM': 'RPM',
            'Pump Type': 'Pump'
        })
        df = df.drop(columns=['Age'])

        X = df[['Height', 'Weight', 'BSA', 'Flow', 'Pump']].values
        y = df[['RPM']].values

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y)

        new_model = Sequential([
            Dense(20, input_dim=5, activation='relu'),
            Dense(20, activation='relu'),
            Dense(1, activation='linear')
        ])
        new_model.compile(optimizer='adam', loss='mean_squared_error')

        callbacks = []
        if early_stop_enabled:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)

        history = new_model.fit(
            X_scaled, y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Загружаем ВСЕ предыдущие модели
        previous_models = self.load_all_models()

        if previous_models:
            # Создаем усреднение со всеми предыдущими моделями
            avg_weights = []
            num_models = len(previous_models) + 1  # +1 для новой модели

            # Инициализируем суммарные веса
            for layer_weights in new_model.get_weights():
                avg_weights.append(np.zeros_like(layer_weights))

            # Суммируем веса всех моделей
            for model in previous_models:
                model_weights = model.get_weights()
                for i in range(len(avg_weights)):
                    avg_weights[i] += model_weights[i]

            # Добавляем веса новой модели
            new_model_weights = new_model.get_weights()
            for i in range(len(avg_weights)):
                avg_weights[i] += new_model_weights[i]

            # Усредняем
            for i in range(len(avg_weights)):
                avg_weights[i] /= num_models

            # Устанавливаем усредненные веса
            new_model.set_weights(avg_weights)
        else:
            # Если это первая модель после базовой, усредняем с базовой
            base_model = load_model("models/initial_model.keras", compile=False)
            new_weights = []
            for new_layer, base_layer in zip(new_model.get_weights(), base_model.get_weights()):
                new_weights.append((new_layer + base_layer) / 2)
            new_model.set_weights(new_weights)

        y_pred = new_model.predict(X_scaled)
        mae = mean_absolute_error(
            y_scaler.inverse_transform(y_scaled),
            y_scaler.inverse_transform(y_pred)
        )

        version_id = f"model_{str(uuid.uuid4())[:8]}.keras"
        model_path = os.path.join("models", version_id)
        save_model(new_model, model_path)

        scaler_path = self.get_scaler_path_for_model(version_id)
        with open(scaler_path, "wb") as f:
            pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        val_loss = min(history.history['val_loss']) if 'val_loss' in history.history and history.history[
            'val_loss'] else 0
        train_loss = min(history.history['loss']) if history.history['loss'] else 0

        with open(self.versions_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{version_id},{created_at},{mae:.6f},{val_loss:.6f},{train_loss:.6f},{scaler_path}\n")

        return version_id, mae, val_loss, scaler_path

    def get_available_models(self):
        keras_files = [f for f in os.listdir("models") if f.endswith(".keras")]
        return sorted(keras_files)


    def get_version_history(self):
        if not os.path.exists(self.versions_file):
            return []

        try:
            with open(self.versions_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return lines
        except:
            return []

    def delete_model(self, model_name):
        """Удаляет модель, связанный с ней scaler и запись из истории версий"""
        try:
            # Удаляем файл модели
            model_path = os.path.join("models", model_name)
            if os.path.exists(model_path):
                os.remove(model_path)

            # Удаляем файл scaler
            scaler_path = self.get_scaler_path_for_model(model_name)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)

            # Удаляем запись из файла версий
            if os.path.exists(self.versions_file):
                with open(self.versions_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = [line for line in lines if not line.startswith(model_name + ",")]

                with open(self.versions_file, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(new_lines)

            return True
        except Exception as e:
            print(f"Ошибка при удалении модели: {str(e)}")
            return False

    def load_all_models(self, exclude_model=None):
        """Загружает все модели кроме указанной"""
        models = []
        model_files = [f for f in os.listdir("models") if f.endswith(".keras")]

        for model_file in model_files:
            if model_file == exclude_model:
                continue
            model_path = os.path.join("models", model_file)
            try:
                model = load_model(model_path, compile=False)
                models.append(model)
            except:
                continue

        return models

