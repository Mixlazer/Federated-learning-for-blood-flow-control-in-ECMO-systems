import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from backend import OxygenBackend

# Отключаем ненужные компоненты TensorFlow
os.environ['TF_DISABLE_TENSORRT'] = '1'  # Отключаем TensorRT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем логи TensorFlow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Принудительный импорт только необходимых модулей
try:
    import tensorflow.python.data.ops.shuffle_op
    import tensorflow.python.data.ops.dataset_ops
    import tensorflow.compiler.tf2tensorrt.ops
except ImportError:
    pass

# Принудительный импорт проблемных модулей
try:
    from tensorflow.python.data.ops import shuffle_op, dataset_ops
except ImportError:
    pass


def get_appdata_path():
    """Возвращает путь для логов"""
    if getattr(sys, 'frozen', False):
        return Path(os.getenv('LOCALAPPDATA')) / 'OxygenationApp' / 'logs'
    else:
        return Path(os.getenv('LOCALAPPDATA')) / 'OxygenationAppDev' / 'logs'


# Настройка логгера
log_dir = get_appdata_path()
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OxygenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blood Oxygenation RPM Predictor")
        self.root.geometry("900x900")

        # Установка обработчика необработанных исключений
        sys.excepthook = self.handle_uncaught_exception

        # Инициализация переменных
        self._init_variables()

        # Инициализация backend
        try:
            self.backend = OxygenBackend()
            logger.info("Backend initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize backend: {str(e)}")
            messagebox.showerror(
                "Critical Error",
                f"Failed to initialize application:\n\n{str(e)}\n\n"
                "Please check the log files and reinstall the application."
            )
            self.root.destroy()
            return

        # Построение интерфейса
        self._build_ui()
        self._build_settings_tab()
        self.update_version_history()

        # Проверка наличия базовой модели
        self._check_base_model()
        self._setup_model_comboboxes()

    def _setup_model_comboboxes(self):
        """Инициализация выпадающих списков моделей"""
        models = self.backend.get_available_models()
        self.model_selector['values'] = models
        self.delete_model_selector['values'] = models

        if models:
            self.selected_model_var.set(models[0])
            self.model_to_delete_var.set(models[0])

    def _init_variables(self):
        """Инициализация переменных Tkinter"""
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar(value="male")
        self.height_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.pump_type_var = tk.StringVar(value="type1")
        self.selected_model_var = tk.StringVar(value="initial_model.keras")
        self.model_to_delete_var = tk.StringVar()

        self.validation_split = tk.DoubleVar(value=0.2)
        self.epochs = tk.IntVar(value=40)
        self.batch_size = tk.IntVar(value=16)
        self.early_stop_enabled = tk.BooleanVar(value=True)
        self.settings_visible = tk.BooleanVar(value=False)

    def _build_ui(self):
        """Построение основного интерфейса"""
        padding = {"padx": 10, "pady": 5}

        # Header frame
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            header_frame,
            text="Расчет оптимальных оборотов насоса",
            font=("Arial", 12, "bold")
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            header_frame,
            text="Настройки обучения",
            command=self.toggle_settings,
            bg="#e0e0ff",
            relief=tk.RAISED
        ).pack(side=tk.RIGHT, padx=10)

        # Input fields frame
        input_frame = tk.LabelFrame(
            self.root,
            text="Параметры пациента",
            padx=10,
            pady=10
        )
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        # Age
        tk.Label(input_frame, text="Возраст").grid(row=0, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.age_var).grid(row=0, column=1, sticky="ew", **padding)

        # Gender
        tk.Label(input_frame, text="Пол").grid(row=1, column=0, sticky="w", **padding)
        ttk.Combobox(
            input_frame,
            values=["male", "female"],
            textvariable=self.gender_var,
            state="readonly",
            width=17
        ).grid(row=1, column=1, sticky="w", **padding)

        # Height
        tk.Label(input_frame, text="Рост (см)").grid(row=2, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.height_var).grid(row=2, column=1, sticky="ew", **padding)

        # Weight
        tk.Label(input_frame, text="Вес (кг)").grid(row=3, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.weight_var).grid(row=3, column=1, sticky="ew", **padding)

        # Pump type
        tk.Label(input_frame, text="Тип насоса").grid(row=4, column=0, sticky="w", **padding)
        ttk.Combobox(
            input_frame,
            values=["type1", "type2", "type3"],
            textvariable=self.pump_type_var,
            state="readonly",
            width=17
        ).grid(row=4, column=1, sticky="w", **padding)

        # Model selection and calculation frame
        model_frame = tk.LabelFrame(
            self.root,
            text="Прогнозирование",
            padx=10,
            pady=10
        )
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model selection
        tk.Label(model_frame, text="Выберите модель:").grid(row=0, column=0, sticky="w", **padding)
        self.model_selector = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model_var,
            state="readonly",
            width=25
        )
        self.model_selector.grid(row=0, column=1, sticky="ew", **padding)

        # Calculate button
        tk.Button(
            model_frame,
            text="Рассчитать RPM",
            command=self.calculate_rpm,
            bg="#d0f0d0",
            height=2
        ).grid(row=0, column=2, padx=10)

        # Result display
        self.result_label = tk.Label(
            model_frame,
            text="",
            font=("Arial", 14),
            fg="blue"
        )
        self.result_label.grid(row=1, column=0, columnspan=3, pady=10)

        # Training frame
        training_frame = tk.LabelFrame(
            self.root,
            text="Дообучение модели",
            padx=10,
            pady=10
        )
        training_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            training_frame,
            text="Загрузить данные и дообучить модель",
            command=self.load_and_fedavg_model,
            bg="#f0d0f0",
            height=2
        ).pack(pady=5)

        # Version history frame
        history_frame = tk.LabelFrame(
            self.root,
            text="История версий моделей",
            padx=10,
            pady=10
        )
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # History controls frame
        history_controls_frame = tk.Frame(history_frame)
        history_controls_frame.pack(fill=tk.X, pady=5)

        # Model to delete selector
        tk.Label(
            history_controls_frame,
            text="Модель для удаления:"
        ).pack(side=tk.LEFT, padx=5)

        self.delete_model_selector = ttk.Combobox(
            history_controls_frame,
            textvariable=self.model_to_delete_var,
            state="readonly",
            width=25
        )
        self.delete_model_selector.pack(side=tk.LEFT, padx=5)

        # Delete button
        tk.Button(
            history_controls_frame,
            text="Удалить модель",
            command=self.delete_selected_model,
            bg="#ffd0d0",
            height=2
        ).pack(side=tk.LEFT, padx=5)

        # MLflow UI button
        tk.Button(
            history_controls_frame,
            text="Open MLflow UI",
            command=self.open_mlflow_ui,
            bg="#d0d0ff",
            height=2
        ).pack(side=tk.LEFT, padx=5)

        # Version history text box
        self.version_box = tk.Text(
            history_frame,
            height=15,
            width=120,
            wrap=tk.NONE
        )
        self.version_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(
            history_frame,
            command=self.version_box.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.version_box.config(yscrollcommand=scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(
            history_frame,
            orient=tk.HORIZONTAL,
            command=self.version_box.xview
        )
        h_scrollbar.pack(fill=tk.X, padx=5)
        self.version_box.config(xscrollcommand=h_scrollbar.set)

    def _build_settings_tab(self):
        """Построение панели настроек обучения"""
        self.settings_frame = tk.LabelFrame(
            self.root,
            text="Параметры обучения",
            padx=10,
            pady=10,
            bd=1,
            relief=tk.SUNKEN
        )

        # Validation split
        validation_frame = tk.Frame(self.settings_frame)
        validation_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            validation_frame,
            text="Доля валидационных данных:",
            width=25,
            anchor="w"
        ).pack(side=tk.LEFT)

        tk.Scale(
            validation_frame,
            from_=0.05,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.validation_split,
            showvalue=True,
            length=300
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Epochs
        epochs_frame = tk.Frame(self.settings_frame)
        epochs_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            epochs_frame,
            text="Количество эпох:",
            width=25,
            anchor="w"
        ).pack(side=tk.LEFT)

        tk.Entry(
            epochs_frame,
            textvariable=self.epochs,
            width=10
        ).pack(side=tk.LEFT)

        # Batch size
        batch_frame = tk.Frame(self.settings_frame)
        batch_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            batch_frame,
            text="Размер батча:",
            width=25,
            anchor="w"
        ).pack(side=tk.LEFT)

        tk.Entry(
            batch_frame,
            textvariable=self.batch_size,
            width=10
        ).pack(side=tk.LEFT)

        # Early stopping
        early_stop_frame = tk.Frame(self.settings_frame)
        early_stop_frame.pack(fill=tk.X, pady=5)

        tk.Checkbutton(
            early_stop_frame,
            text="Ранняя остановка при отсутствии улучшений",
            variable=self.early_stop_enabled,
            anchor="w",
            width=35
        ).pack(side=tk.LEFT)

    def _check_base_model(self):
        """Проверка наличия базовой модели"""
        try:
            models = self.backend.get_available_models()
            if "initial_model.keras" not in models:
                raise FileNotFoundError("Base model not found")
        except Exception as e:
            logger.error(f"Base model check failed: {str(e)}")
            messagebox.showerror(
                "Critical Error",
                "Base model files are missing. Please reinstall the application."
            )
            self.root.destroy()

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Обработчик необработанных исключений"""
        logger.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        messagebox.showerror(
            "Critical Error",
            f"An unexpected error occurred:\n\n{str(exc_value)}\n\n"
            "Please check the log files for more details."
        )

        self.root.destroy()

    def toggle_settings(self):
        """Переключение видимости панели настроек"""
        if self.settings_visible.get():
            self.settings_frame.pack_forget()
            self.settings_visible.set(False)
        else:
            self.settings_frame.pack(
                fill=tk.X,
                padx=10,
                pady=5,
                before=self.root.winfo_children()[-2]
            )
            self.settings_visible.set(True)

    def open_mlflow_ui(self):
        """Открытие UI MLflow в браузере"""
        try:
            result = self.backend.open_mlflow_ui()
            messagebox.showinfo(
                "MLflow UI",
                result
            )
        except Exception as e:
            logger.error(f"Failed to open MLflow UI: {str(e)}")
            messagebox.showerror(
                "Ошибка",
                f"Не удалось запустить MLflow UI:\n\n{str(e)}"
            )

    def calculate_rpm(self):
        """Расчет оптимальных оборотов насоса"""
        try:
            # Проверка ввода
            if not self.height_var.get() or not self.weight_var.get():
                raise ValueError("Введите рост и вес пациента")

            height = float(self.height_var.get())
            weight = float(self.weight_var.get())

            if height <= 0 or weight <= 0:
                raise ValueError("Рост и вес должны быть положительными числами")

            pump_type = self.pump_type_var.get()
            model_name = self.selected_model_var.get()

            # Выполнение расчета
            rpm = self.backend.calculate_rpm(height, weight, pump_type, model_name)

            # Отображение результата
            self.result_label.config(
                text=f"Оптимальные обороты: {rpm:.2f} RPM",
                fg="green"
            )

        except ValueError as e:
            self.result_label.config(
                text=f"Ошибка ввода: {str(e)}",
                fg="red"
            )
        except Exception as e:
            self.result_label.config(
                text=f"Ошибка расчета: {str(e)}",
                fg="red"
            )
            logger.error(f"Calculation error: {str(e)}")

    def load_and_fedavg_model(self):
        """Загрузка данных и дообучение модели"""
        try:
            # Запрос файла с данными
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )

            if not file_path:
                logger.info("Model training canceled - no file selected")
                return

            # Проверка параметров обучения
            if not 0 < self.validation_split.get() < 1:
                raise ValueError("Validation split must be between 0 and 1")
            if self.epochs.get() <= 0:
                raise ValueError("Number of epochs must be positive")
            if self.batch_size.get() <= 0:
                raise ValueError("Batch size must be positive")

            # Выполнение дообучения
            version_id, mae, val_loss, scaler_path = self.backend.load_and_fedavg_model(
                file_path,
                self.validation_split.get(),
                self.epochs.get(),
                self.batch_size.get(),
                self.early_stop_enabled.get()
            )

            # Отображение результатов
            messagebox.showinfo(
                "Успех",
                f"Модель успешно дообучена и сохранена как:\n{version_id}\n\n"
                f"Средняя абсолютная ошибка (MAE): {mae:.2f}\n"
                f"Минимальная потеря на валидации: {val_loss:.4f}"
            )

            # Обновление истории версий
            self.update_version_history()

            logger.info(f"Model training completed successfully. New model: {version_id}")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            messagebox.showerror(
                "Ошибка",
                f"Не удалось дообучить модель:\n\n{str(e)}"
            )

    def delete_selected_model(self):
        """Удаление выбранной модели"""
        model_name = self.model_to_delete_var.get()

        if not model_name:
            messagebox.showwarning(
                "Предупреждение",
                "Не выбрана модель для удаления"
            )
            return

        if model_name == "initial_model.keras":
            messagebox.showwarning(
                "Предупреждение",
                "Нельзя удалить базовую модель"
            )
            return

        # Подтверждение удаления
        confirm = messagebox.askyesno(
            "Подтверждение",
            f"Вы уверены, что хотите удалить модель {model_name}?\n"
            "Это действие нельзя отменить."
        )

        if not confirm:
            return

        try:
            # Выполнение удаления
            success = self.backend.delete_model(model_name)

            if success:
                messagebox.showinfo(
                    "Успех",
                    f"Модель {model_name} успешно удалена"
                )

                # Обновление интерфейса
                self.update_version_history()
                logger.info(f"Successfully deleted model: {model_name}")
            else:
                raise Exception("Unknown error occurred during deletion")

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {str(e)}")
            messagebox.showerror(
                "Ошибка",
                f"Не удалось удалить модель {model_name}:\n\n{str(e)}"
            )

    def update_version_history(self):
        """Обновление истории версий и выпадающих списков"""
        try:
            self.version_box.delete("1.0", tk.END)

            # Получаем актуальный список моделей
            models = self.backend.get_available_models()

            # Обновляем выпадающие списки
            self.model_selector['values'] = models
            self.delete_model_selector['values'] = models

            # Устанавливаем значения по умолчанию
            if models:
                current_model = self.selected_model_var.get()
                delete_model = self.model_to_delete_var.get()

                if not current_model or current_model not in models:
                    self.selected_model_var.set(models[0])
                if not delete_model or delete_model not in models:
                    self.model_to_delete_var.set(models[0])
            else:
                self.selected_model_var.set("")
                self.model_to_delete_var.set("")

            # Отображаем историю версий
            lines = self.backend.get_version_history()
            if not lines:
                self.version_box.insert(tk.END, "История версий пуста.\n")
                return

            # Заголовок таблицы
            headers = ["version_id", "created_at", "mae", "val_loss", "train_loss"]
            header_text = f"{headers[0]:<20} {headers[1]:<20} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}\n"
            self.version_box.insert(tk.END, header_text)
            self.version_box.insert(tk.END, "-" * 80 + "\n")

            # Данные
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    row_text = f"{parts[0]:<20} {parts[1]:<20} {parts[2]:<10} {parts[3]:<10} {parts[4]:<10}\n"
                    self.version_box.insert(tk.END, row_text)

        except Exception as e:
            logger.error(f"Error updating version history: {str(e)}")
            messagebox.showerror("Ошибка", "Не удалось обновить историю версий")


if __name__ == "__main__":
    root = tk.Tk()
    app = OxygenApp(root)
    root.mainloop()