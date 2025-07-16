import os
import uuid
import numpy as np
import pandas as pd
import math
import pickle
import logging
import sys
import tensorflow as tf
from keras.models import load_model, save_model, Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
import mlflow
import mlflow.tensorflow
import subprocess
import webbrowser
import sqlite3
from mlflow.models.signature import infer_signature

# Отключение интерактивного логирования TensorFlow
tf.keras.utils.disable_interactive_logging()


def get_appdata_path():
    """Возвращает корректный путь для хранения данных в AppData/Local"""
    if getattr(sys, 'frozen', False):
        # Для собранного .exe
        base_path = Path(os.getenv('LOCALAPPDATA')) / 'OxygenationApp'
    else:
        # Для разработки
        base_path = Path(os.getenv('LOCALAPPDATA')) / 'OxygenationAppDev'

    # Создаем необходимые папки
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


# Инициализация логгера
base_path = get_appdata_path()
log_file = base_path / 'logs' / 'backend.log'
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OxygenBackend:
    def __init__(self):
        self.data_dir = get_appdata_path()
        self.models_dir = self.data_dir / 'models'
        self.versions_file = self.models_dir / 'versions.csv'

        # Создаем папку моделей, если не существует
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Инициализация файла версий
        self._init_versions_file()

        # Проверяем наличие моделей
        if not any(self.models_dir.glob('*.keras')):
            self._copy_default_models()

        # Инициализация MLflow
        self.mlflow_enabled = True
        self._init_mlflow()

    def _init_mlflow(self):
        """Инициализация MLflow с использованием SQLite"""
        try:
            # Путь к файлу mlflow.db
            self.mlflow_db_path = self.data_dir / "mlflow.db"
            self.mlflow_rundir = self.data_dir / "mlruns"
            self.mlflow_rundir.mkdir(exist_ok=True)

            # Настраиваем URI
            tracking_uri = f"sqlite:///{self.mlflow_db_path.as_posix()}"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

            # Если эксперимент ещё не создан — создаём его с явно указанным местом для артефактов
            exp = mlflow.get_experiment_by_name("OxygenationRPM")
            if exp is None:
                mlflow.create_experiment(
                    name="OxygenationRPM",
                    artifact_location=self.mlflow_rundir.as_posix()
                )
                logger.info(f"Created MLflow experiment at {self.mlflow_rundir}")
            # Теперь просто переключаемся на него
            mlflow.set_experiment("OxygenationRPM")
            logger.info("MLflow initialized successfully")

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}", exc_info=True)
            # Отключаем все вызовы MLflow
            self.mlflow_enabled = False

    def open_mlflow_ui(self):
        """Запуск UI MLflow с обработкой ошибок"""
        if not self.mlflow_enabled:
            return "MLflow не инициализирован. Проверьте логи для подробностей."

        try:
            store_uri = f"sqlite:///{self.mlflow_db_path.as_posix()}"

            # Запускаем MLflow UI
            subprocess.Popen(
                ["mlflow", "ui", "--backend-store-uri", store_uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )

            # Открываем в браузере через 2 секунды
            import threading
            def open_browser():
                import time
                time.sleep(2)
                webbrowser.open("http://localhost:5000")

            threading.Thread(target=open_browser, daemon=True).start()

            return "MLflow UI запущен. Откройте http://localhost:5000"

        except Exception as e:
            logger.error(f"Failed to start MLflow UI: {str(e)}")
            return f"Ошибка запуска MLflow UI: {str(e)}"

    def _init_models_dir(self):
        """Инициализирует папку с моделями"""
        models_path = self.data_dir / "models"
        os.makedirs(models_path, exist_ok=True)
        return models_path

    def _copy_default_models(self):
        """Копирует модели из папки установки в AppData"""
        try:
            # Получаем путь к исходным моделям
            if getattr(sys, 'frozen', False):
                src_dir = Path(getattr(sys, '_MEIPASS', Path.cwd())) / 'models'
            else:
                src_dir = Path(__file__).parent / 'models'

            logger.info(f"Copying models from {src_dir} to {self.models_dir}")

            # Копируем все файлы из исходной директории
            for file_path in src_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, self.models_dir)

            logger.info("Default models copied successfully")
        except Exception as e:
            logger.exception(f"Failed to copy default models: {str(e)}")
            raise RuntimeError(f"Failed to initialize application: {str(e)}")

    def _init_versions_file(self):
        """Инициализация файла версий с обработкой ошибок"""
        try:
            if not os.path.exists(self.versions_file):
                with open(self.versions_file, "w", encoding="utf-8", newline="\n") as f:
                    f.write("version_id,created_at,mae,val_loss,train_loss\n")
        except Exception as e:
            logger.error(f"Error initializing versions file: {str(e)}")
            # Попробуем создать в временной папке
            temp_versions = os.path.join(tempfile.gettempdir(), "oxygen_versions.csv")
            logger.warning(f"Using temporary versions file: {temp_versions}")
            self.versions_file = temp_versions
            with open(self.versions_file, "w", encoding="utf-8", newline="\n") as f:
                f.write("version_id,created_at,mae,val_loss,train_loss\n")

    def ensure_initial_model(self):
        """Проверка наличия базовой модели"""
        initial_model_path = self.models_dir / "initial_model.keras"
        initial_scaler_path = self.models_dir / "initial_model.scaler"

        if not os.path.exists(initial_model_path) or not os.path.exists(initial_scaler_path):
            logger.error(
                f"Initial model or scaler not found. Model: {initial_model_path}, Scaler: {initial_scaler_path}")
            raise FileNotFoundError(
                "Base model files not found. Please reinstall the application with all required model files."
            )

    def calculate_rpm(self, height, weight, pump_type, model_name):
        """Расчет оптимальных оборотов насоса"""
        try:
            logger.info(f"Calculating RPM for height={height}, weight={weight}, pump={pump_type}, model={model_name}")

            # Проверка входных данных
            if not all(isinstance(x, (int, float)) for x in [height, weight]):
                raise ValueError("Height and weight must be numeric values")

            pump_map = {"type1": 1, "type2": 2, "type3": 3}
            pump = pump_map.get(pump_type, 1)
            bsa = math.sqrt(height * weight / 3600)
            flow = bsa * 2.4
            X_new = np.array([[height, weight, bsa, flow, pump]])

            # Получение абсолютных путей к файлам
            model_path = self.models_dir / model_name
            scaler_path = self.get_scaler_path_for_model(model_name)

            logger.debug(f"Model path: {model_path}")
            logger.debug(f"Scaler path: {scaler_path}")

            # Проверка существования файлов
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

            # Загрузка скейлера с проверкой
            try:
                with open(scaler_path, "rb") as f:
                    scaler_data = pickle.load(f)
                    if not all(key in scaler_data for key in ['x_scaler', 'y_scaler']):
                        raise ValueError("Invalid scaler file format")
                    x_scaler = scaler_data['x_scaler']
                    y_scaler = scaler_data['y_scaler']
            except Exception as e:
                raise ValueError(f"Failed to load scaler: {str(e)}")

            # Масштабирование входных данных
            X_new_scaled = x_scaler.transform(X_new)

            # Загрузка модели с проверкой
            try:
                model = load_model(model_path)
                if not hasattr(model, 'predict'):
                    raise ValueError("Loaded model is invalid")
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")

            # Предсказание
            try:
                rpm_scaled = model.predict(X_new_scaled)
                rpm = y_scaler.inverse_transform(rpm_scaled)[0][0]
            except Exception as e:
                raise ValueError(f"Prediction failed: {str(e)}")

            logger.info(f"Successfully calculated RPM: {rpm:.2f}")
            return rpm

        except Exception as e:
            logger.exception("Error in calculate_rpm")
            raise ValueError(f"Calculation error: {str(e)}")

    def get_scaler_path_for_model(self, model_name):
        """Полный путь к файлу скейлера"""
        base_name = os.path.splitext(model_name)[0]
        return self.models_dir / f"{base_name}.scaler"

    def load_and_fedavg_model(self, file_path, validation_split, epochs, batch_size, early_stop_enabled):
        """Улучшенная версия Federated Averaging с реальным усреднением"""
        try:
            # Загрузка данных (без изменений)
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Подготовка данных (без изменений)
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

            # Масштабирование
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y)

            # 1. Загрузка ВСЕХ предыдущих моделей
            previous_models = self.load_all_models()
            logger.info(f"Loaded {len(previous_models)} previous models for averaging")

            # 2. Создание новой модели с такой же архитектурой
            new_model = Sequential([
                Dense(20, input_dim=5, activation='relu'),
                Dense(20, activation='relu'),
                Dense(1, activation='linear')
            ])
            new_model.compile(optimizer='adam', loss='mean_squared_error')

            # 3. Обучение новой модели
            history = new_model.fit(
                X_scaled, y_scaled,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(patience=5)] if early_stop_enabled else [],
                verbose=1
            )

            # 4. Federated Averaging с улучшенной логикой
            if previous_models:
                logger.info("Starting Federated Averaging process...")

                # Инициализация средних весов
                avg_weights = [np.zeros_like(w) for w in new_model.get_weights()]
                total_samples = len(X_scaled)  # Вес новой модели

                # Учет весов новой модели
                for i, w in enumerate(new_model.get_weights()):
                    avg_weights[i] += w * total_samples

                # Собираем веса всех моделей
                for model in previous_models:
                    model_weights = model.get_weights()
                    # Предполагаем, что каждая предыдущая модель была обучена на таком же объеме данных
                    for i in range(len(avg_weights)):
                        avg_weights[i] += model_weights[i] * total_samples

                # Усреднение
                num_models = len(previous_models) + 1  # + новая модель
                for i in range(len(avg_weights)):
                    avg_weights[i] /= (num_models * total_samples)

                # Применяем усредненные веса
                new_model.set_weights(avg_weights)
                logger.info(f"Successfully averaged weights from {num_models} models")

            # 5. Повторное обучение с усредненными весами (fine-tuning)
            if previous_models:  # Только если были предыдущие модели
                logger.info("Fine-tuning averaged model...")
                history_fine = new_model.fit(
                    X_scaled, y_scaled,
                    validation_split=validation_split,
                    epochs=max(epochs // 2, 5),  # Уменьшенное количество эпох
                    batch_size=batch_size,
                    verbose=1
                )
                # Объединяем историю обучения
                history.history['loss'] += history_fine.history['loss']
                history.history['val_loss'] += history_fine.history['val_loss']

            # Оценка модели
            y_pred = new_model.predict(X_scaled)
            mae = mean_absolute_error(
                y_scaler.inverse_transform(y_scaled),
                y_scaler.inverse_transform(y_pred)
            )

            # Сохранение модели и скейлера
            version_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            model_path = self.models_dir / version_id
            save_model(new_model, model_path)

            scaler_path = self.get_scaler_path_for_model(version_id)
            with open(scaler_path, "wb") as f:
                pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

            # Логирование в MLflow (если нужно)
            if self.mlflow_enabled:
                with mlflow.start_run():
                    mlflow.log_params({
                        "validation_split": validation_split,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "early_stop": early_stop_enabled,
                        "fedavg_models": len(previous_models)
                    })
                    mlflow.log_metrics({
                        "mae": mae,
                        "val_loss": min(history.history['val_loss']),
                        "train_loss": min(history.history['loss'])
                    })
                    mlflow.log_metric("num_models_avg", len(previous_models))

            # Запись в файл версий
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            val_loss = min(history.history['val_loss'])
            train_loss = min(history.history['loss'])

            self._append_to_versions_file(
                version_id, created_at, mae, val_loss, train_loss
            )

            logger.info(f"Model training completed. MAE: {mae:.2f}")
            return version_id, mae, val_loss, scaler_path

        except Exception as e:
            logger.exception("Error in load_and_fedavg_model")
            raise

    def _append_to_versions_file(self, version_id, created_at, mae, val_loss, train_loss):
        """Безопасное добавление записи в файл версий"""
        try:
            with open(self.versions_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"{version_id},{created_at},{mae:.6f},{val_loss:.6f},{train_loss:.6f}\n")
        except Exception as e:
            logger.error(f"Failed to write to versions file: {str(e)}")
            # Создаем резервную копию
            backup_file = f"{self.versions_file}.backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            with open(backup_file, "w", encoding="utf-8", newline="\n") as backup:
                backup.write(f"{version_id},{created_at},{mae:.6f},{val_loss:.6f},{train_loss:.6f}\n")
            logger.info(f"Created backup versions file: {backup_file}")

    def _get_models_dir(self):
        """Определяет правильный путь к папке models"""
        return self.models_dir

    def _fix_model_names_case(self):
        """Исправляет регистр имен файлов моделей"""
        for f in os.listdir(self.models_dir):
            if f.lower() == 'initial_model.keras' and f != 'initial_model.keras':
                old_path = os.path.join(self.models_dir, f)
                new_path = os.path.join(self.models_dir, 'initial_model.keras')
                os.rename(old_path, new_path)

    def get_available_models(self):
        """Получение списка моделей"""
        try:
            # Проверка существования директории
            if not os.path.exists(self.models_dir):
                return ['initial_model.keras']

            models = []
            for f in os.listdir(self.models_dir):
                if f.endswith('.keras'):
                    # Для базовой модели сохраняем в нижнем регистре
                    if f.lower() == 'initial_model.keras':
                        models.append('initial_model.keras')
                    else:
                        models.append(f)

            # Убедимся, что базовая модель всегда первая
            sorted_models = []
            if 'initial_model.keras' in models:
                sorted_models.append('initial_model.keras')
                models.remove('initial_model.keras')
            sorted_models.extend(sorted(models))
            return sorted_models
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return ['initial_model.keras']

    def get_version_history(self):
        """Получение истории версий моделей"""
        try:
            if not os.path.exists(self.versions_file):
                return []

            with open(self.versions_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            return lines
        except Exception as e:
            logger.error(f"Error reading version history: {str(e)}")
            return []

    def delete_model(self, model_name):
        """Удаление модели с учетом регистра"""
        try:
            logger.info(f"Deleting model: {model_name}")

            # Приводим имя файла к правильному регистру
            actual_model_name = "initial_model.keras" if model_name.lower() == "initial_model.keras" else model_name

            model_path = self.models_dir / actual_model_name
            scaler_path = self.get_scaler_path_for_model(actual_model_name)

            # Удаление файлов
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)

            # Обновление файла версий
            if os.path.exists(self.versions_file):
                with open(self.versions_file, "r", encoding="utf-8") as f:
                    lines = [line for line in f.readlines() if not line.startswith(actual_model_name + ",")]

                with open(self.versions_file, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(lines)

            return True
        except Exception as e:
            logger.exception(f"Error deleting model {model_name}")
            raise ValueError(f"Failed to delete model: {str(e)}")

    def load_all_models(self, exclude_model=None):
        """Загрузка всех моделей с проверкой архитектуры"""
        models = []
        try:
            model_files = [f for f in os.listdir(self.models_dir)
                           if f.endswith(".keras") and f != exclude_model]

            # Сортируем по дате создания (сначала новые)
            model_files.sort(key=lambda x: os.path.getmtime(self.models_dir / x), reverse=True)

            # Ограничиваем количество моделей для усреднения (опционально)
            max_models = 10  # Можно настроить
            model_files = model_files[:max_models]

            for model_file in model_files:
                try:
                    model = load_model(self.models_dir / model_file, compile=False)

                    # Проверяем, что архитектура совместима
                    if len(model.layers) == 3:  # Должно соответствовать новой модели
                        models.append(model)
                        logger.debug(f"Loaded model: {model_file}")
                    else:
                        logger.warning(f"Skipping incompatible model: {model_file}")

                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {str(e)}")
                    continue

            return models
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return []