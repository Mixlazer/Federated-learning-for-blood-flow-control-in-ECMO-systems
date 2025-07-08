import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import requests

class OxygenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Blood Oxygenation System")
        self.root.geometry("520x620")

        # Переменные для ввода
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar(value="male")
        self.height_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.pump_type_var = tk.StringVar(value="type1")

        self.build_ui()

    def build_ui(self):
        padding = {"padx": 10, "pady": 5}

        # Вводные поля
        tk.Label(self.root, text="Возраст").pack(**padding)
        tk.Entry(self.root, textvariable=self.age_var).pack(**padding)

        tk.Label(self.root, text="Пол").pack(**padding)
        gender_combo = ttk.Combobox(self.root, values=["male", "female"], textvariable=self.gender_var, state="readonly")
        gender_combo.pack(**padding)

        tk.Label(self.root, text="Рост (см)").pack(**padding)
        tk.Entry(self.root, textvariable=self.height_var).pack(**padding)

        tk.Label(self.root, text="Вес (кг)").pack(**padding)
        tk.Entry(self.root, textvariable=self.weight_var).pack(**padding)

        tk.Label(self.root, text="Тип насоса").pack(**padding)
        pump_combo = ttk.Combobox(self.root, values=["type1", "type2", "type3"], textvariable=self.pump_type_var, state="readonly")
        pump_combo.pack(**padding)

        # Кнопка расчета в центре
        calc_btn = tk.Button(self.root, text="Рассчитать RPM", command=self.calculate_rpm)
        calc_btn.pack(pady=15)

        # Метка для результата
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=10)

        # Панель с дополнительными кнопками (настройки, загрузка данных, история версий)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10, fill=tk.X)

        # Кнопка Настройки (справа вверху)
        settings_btn = tk.Button(btn_frame, text="Настройки", command=self.show_settings)
        settings_btn.pack(side=tk.RIGHT, padx=10)

        # Кнопка Загрузить данные для дообучения
        upload_btn = tk.Button(btn_frame, text="Загрузить данные для дообучения", command=self.upload_data)
        upload_btn.pack(side=tk.RIGHT, padx=10)

        # Кнопка История версий
        versions_btn = tk.Button(btn_frame, text="История версий", command=self.show_versions)
        versions_btn.pack(side=tk.RIGHT, padx=10)

        # Текстовое окно для отображения истории версий
        self.version_box = tk.Text(self.root, height=12, width=60)
        self.version_box.pack(pady=10)

    def calculate_rpm(self):
        try:
            data = {
                "age": int(self.age_var.get()),
                "gender": self.gender_var.get(),
                "height": float(self.height_var.get()),
                "weight": float(self.weight_var.get()),
                "pump_type": self.pump_type_var.get()
            }
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные данные")
            return

        threading.Thread(target=self._send_predict, args=(data,)).start()

    def _send_predict(self, data):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=data)
            response.raise_for_status()
            rpm = response.json().get("predicted_rpm", "-")
            self.result_label.config(text=f"Предсказанные обороты: {rpm:.2f} RPM")
        except Exception as e:
            messagebox.showerror("Ошибка при запросе", str(e))

    def upload_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")])
        if not file_path:
            return

        threading.Thread(target=self._upload_file_thread, args=(file_path,)).start()

    def _upload_file_thread(self, file_path):
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://127.0.0.1:8000/upload-data", files=files)
                response.raise_for_status()
                messagebox.showinfo("Успех", "Данные загружены. Новая версия модели создана.")
        except Exception as e:
            messagebox.showerror("Ошибка при загрузке данных", str(e))

    def show_versions(self):
        threading.Thread(target=self._fetch_versions).start()

    def _fetch_versions(self):
        try:
            response = requests.get("http://127.0.0.1:8000/versions")
            response.raise_for_status()
            versions = response.json()
            self.version_box.delete("1.0", tk.END)
            if not versions:
                self.version_box.insert(tk.END, "История версий пуста.\n")
            for v in versions:
                line = f"ID: {v['id']} | Run ID: {v['run_id']}\nДата: {v['created_at']} | MSE: {v['mse']:.4f}\n\n"
                self.version_box.insert(tk.END, line)
        except Exception as e:
            messagebox.showerror("Ошибка при получении версий", str(e))

    def show_settings(self):
        # Пример окна настроек — здесь можно расширить по необходимости
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Настройки")
        settings_win.geometry("300x200")
        tk.Label(settings_win, text="Здесь можно добавить настройки приложения").pack(pady=20)
        tk.Button(settings_win, text="Закрыть", command=settings_win.destroy).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = OxygenApp(root)
    root.mainloop()
