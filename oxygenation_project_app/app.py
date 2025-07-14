import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from backend import OxygenBackend


class OxygenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blood Oxygenation RPM Predictor")
        self.root.geometry("900x900")

        # Инициализация всех переменных
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
        self.settings_visible = False

        self.backend = OxygenBackend()

        self.build_ui()
        self.build_settings_tab()
        self.update_version_history()

    def build_ui(self):
        padding = {"padx": 10, "pady": 5}

        # Header with settings button
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill=tk.X, pady=5)

        tk.Label(header_frame, text="Расчет оптимальных оборотов насоса",
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)

        tk.Button(header_frame, text="Настройки обучения", command=self.toggle_settings,
                  bg="#e0e0ff", relief=tk.RAISED).pack(side=tk.RIGHT, padx=10)

        # Input fields
        input_frame = tk.LabelFrame(self.root, text="Параметры пациента", padx=10, pady=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(input_frame, text="Возраст").grid(row=0, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.age_var).grid(row=0, column=1, sticky="ew", **padding)

        tk.Label(input_frame, text="Пол").grid(row=1, column=0, sticky="w", **padding)
        ttk.Combobox(input_frame, values=["male", "female"],
                     textvariable=self.gender_var, state="readonly", width=17).grid(row=1, column=1, sticky="w",
                                                                                    **padding)

        tk.Label(input_frame, text="Рост (см)").grid(row=2, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.height_var).grid(row=2, column=1, sticky="ew", **padding)

        tk.Label(input_frame, text="Вес (кг)").grid(row=3, column=0, sticky="w", **padding)
        tk.Entry(input_frame, textvariable=self.weight_var).grid(row=3, column=1, sticky="ew", **padding)

        tk.Label(input_frame, text="Тип насоса").grid(row=4, column=0, sticky="w", **padding)
        ttk.Combobox(input_frame, values=["type1", "type2", "type3"],
                     textvariable=self.pump_type_var, state="readonly", width=17).grid(row=4, column=1, sticky="w",
                                                                                       **padding)

        # Model selection and calculation
        model_frame = tk.LabelFrame(self.root, text="Прогнозирование", padx=10, pady=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(model_frame, text="Выберите модель:").grid(row=0, column=0, sticky="w", **padding)
        self.model_selector = ttk.Combobox(model_frame, textvariable=self.selected_model_var,
                                           state="readonly", width=25)
        self.model_selector.grid(row=0, column=1, sticky="ew", **padding)

        tk.Button(model_frame, text="Рассчитать RPM", command=self.calculate_rpm,
                  bg="#d0f0d0", height=2).grid(row=0, column=2, padx=10)

        self.result_label = tk.Label(model_frame, text="", font=("Arial", 14), fg="blue")
        self.result_label.grid(row=1, column=0, columnspan=3, pady=10)

        # Training section
        training_frame = tk.LabelFrame(self.root, text="Дообучение модели", padx=10, pady=10)
        training_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(training_frame, text="Загрузить данные и дообучить модель",
                  command=self.load_and_fedavg_model, bg="#f0d0f0", height=2).pack(pady=5)

        # Version history
        history_frame = tk.LabelFrame(self.root, text="История версий моделей", padx=10, pady=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Фрейм для кнопок управления историей
        history_controls_frame = tk.Frame(history_frame)
        history_controls_frame.pack(fill=tk.X, pady=5)

        # Выпадающий список для выбора модели для удаления
        tk.Label(history_controls_frame, text="Модель для удаления:").pack(side=tk.LEFT, padx=5)
        self.delete_model_selector = ttk.Combobox(
            history_controls_frame,
            textvariable=self.model_to_delete_var,
            state="readonly",
            width=25
        )
        self.delete_model_selector.pack(side=tk.LEFT, padx=5)

        # Кнопка удаления
        tk.Button(
            history_controls_frame,
            text="Удалить модель",
            command=self.delete_selected_model,
            bg="#ffd0d0",
            height=2
        ).pack(side=tk.LEFT, padx=5)

        self.version_box = tk.Text(history_frame, height=15, width=120)
        self.version_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(history_frame, command=self.version_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.version_box.config(yscrollcommand=scrollbar.set)

    def build_settings_tab(self):
        self.settings_frame = tk.LabelFrame(self.root, text="Параметры обучения",
                                            padx=10, pady=10, bd=1, relief=tk.SUNKEN)

        # Validation split
        validation_frame = tk.Frame(self.settings_frame)
        validation_frame.pack(fill=tk.X, pady=5)

        tk.Label(validation_frame, text="Доля валидационных данных:", width=25, anchor="w").pack(side=tk.LEFT)
        tk.Scale(validation_frame, from_=0.05, to=0.5, resolution=0.01,
                 orient=tk.HORIZONTAL, variable=self.validation_split,
                 showvalue=True, length=300).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Epochs
        epochs_frame = tk.Frame(self.settings_frame)
        epochs_frame.pack(fill=tk.X, pady=5)

        tk.Label(epochs_frame, text="Количество эпох:", width=25, anchor="w").pack(side=tk.LEFT)
        tk.Entry(epochs_frame, textvariable=self.epochs, width=10).pack(side=tk.LEFT)

        # Batch size
        batch_frame = tk.Frame(self.settings_frame)
        batch_frame.pack(fill=tk.X, pady=5)

        tk.Label(batch_frame, text="Размер батча:", width=25, anchor="w").pack(side=tk.LEFT)
        tk.Entry(batch_frame, textvariable=self.batch_size, width=10).pack(side=tk.LEFT)

        # Early stopping
        early_stop_frame = tk.Frame(self.settings_frame)
        early_stop_frame.pack(fill=tk.X, pady=5)

        tk.Checkbutton(early_stop_frame, text="Ранняя остановка при отсутствии улучшений",
                       variable=self.early_stop_enabled, anchor="w", width=35).pack(side=tk.LEFT)

    def toggle_settings(self):
        if self.settings_visible:
            self.settings_frame.pack_forget()
            self.settings_visible = False
        else:
            self.settings_frame.pack(fill=tk.X, padx=10, pady=5, before=self.root.winfo_children()[-2])
            self.settings_visible = True

    def calculate_rpm(self):
        try:
            height = float(self.height_var.get())
            weight = float(self.weight_var.get())
            pump_type = self.pump_type_var.get()
            model_name = self.selected_model_var.get()

            rpm = self.backend.calculate_rpm(height, weight, pump_type, model_name)
            self.result_label.config(text=f"Предсказанные обороты: {rpm:.2f} RPM")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_and_fedavg_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        if not file_path:
            return

        try:
            version_id, mae, val_loss, scaler_path = self.backend.load_and_fedavg_model(
                file_path,
                self.validation_split.get(),
                self.epochs.get(),
                self.batch_size.get(),
                self.early_stop_enabled.get()
            )
            messagebox.showinfo("Успех",
                                f"Модель успешно дообучена и сохранена как:\n{version_id}\n\n"
                                f"Средняя абсолютная ошибка (MAE): {mae:.2f}\n"
                                f"Минимальная потеря на валидации: {val_loss:.4f}")
            self.update_version_history()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def delete_selected_model(self):
        model_name = self.model_to_delete_var.get()
        if not model_name:
            messagebox.showwarning("Предупреждение", "Не выбрана модель для удаления")
            return

        if model_name == "initial_model.keras":
            messagebox.showwarning("Предупреждение", "Нельзя удалить базовую модель")
            return

        if messagebox.askyesno(
                "Подтверждение",
                f"Вы уверены, что хотите удалить модель {model_name}?\n"
                "Это действие нельзя отменить."
        ):
            success = self.backend.delete_model(model_name)
            if success:
                messagebox.showinfo("Успех", f"Модель {model_name} успешно удалена")
                self.update_version_history()

                # Обновляем выбранные модели в обоих комбобоксах
                models = self.backend.get_available_models()
                if models:
                    self.selected_model_var.set(models[0])
                    self.model_to_delete_var.set(models[0])
                else:
                    self.selected_model_var.set("")
                    self.model_to_delete_var.set("")
            else:
                messagebox.showerror("Ошибка", f"Не удалось удалить модель {model_name}")

    def update_version_history(self):
        self.version_box.delete("1.0", tk.END)

        # Обновляем оба выпадающих списка
        models = self.backend.get_available_models()

        # Для прогнозирования
        self.model_selector["values"] = models
        if not self.selected_model_var.get() and models:
            self.selected_model_var.set(models[0])

        # Для удаления
        self.delete_model_selector["values"] = models
        if not self.model_to_delete_var.get() and models:
            self.model_to_delete_var.set(models[0])

        # Получаем историю версий
        lines = self.backend.get_version_history()
        if not lines:
            self.version_box.insert(tk.END, "История версий пуста.\n")
            return

        try:
            headers = lines[0].strip().split(',')
            headers = ["version_id", "created_at", "mae", "val_loss", "train_loss", "scaler_path"]

            header_text = f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}\n"
            self.version_box.insert(tk.END, header_text)
            self.version_box.insert(tk.END, "-" * 80 + "\n")

            self.version_box.tag_configure("header", font=("Arial", 9, "bold"))
            self.version_box.tag_add("header", "1.0", "1.0 lineend")

            for line in lines[0:]:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                version_id = parts[0]
                created_at = parts[1]
                mae = parts[2] if len(parts) > 2 else "N/A"
                val_loss = parts[3] if len(parts) > 3 else "N/A"
                train_loss = parts[4] if len(parts) > 4 else "N/A"

                version_line = f"{version_id:<15} {created_at:<20} {mae:<10} {val_loss:<10} {train_loss:<10}\n"
                self.version_box.insert(tk.END, version_line)

        except Exception as e:
            self.version_box.insert(tk.END, f"Ошибка загрузки истории: {str(e)}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = OxygenApp(root)
    root.mainloop()