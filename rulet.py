import tkinter as tk
from tkinter import ttk, filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle


class RoulettePredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Программа рулетка")

        self.history = []
        self.model = None
        self.optimization_method = tk.StringVar(value="kNN")
        self.param_values = {
            'n_neighbors': tk.StringVar(value='12'),
            'n_estimators': tk.StringVar(value='100'),
            'max_depth': tk.StringVar(value='10'),
            'C': tk.StringVar(value='1.0')
        }
        self.default_params = {
            'kNN': {'n_neighbors': '12', 'n_estimators': '', 'max_depth': '', 'C': ''},
            'Случайный лес': {'n_neighbors': '12', 'n_estimators': '100', 'max_depth': '10', 'C': ''},
            'Логистическая регрессия': {'n_neighbors': '', 'n_estimators': '', 'max_depth': '', 'C': '1.0'},
            'Наивный байесовский': {'n_neighbors': '', 'n_estimators': '', 'max_depth': '', 'C': ''}
        }

        self.create_widgets()
        self.set_default_params()

    def create_widgets(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=20, pady=20)

        self.label = tk.Label(self.frame, text="Введите число от 0 до 36:")
        self.label.grid(row=0, column=0)

        self.entry = tk.Entry(self.frame)
        self.entry.grid(row=0, column=1)
        self.entry.bind("<Return>", self.on_button_click)

        self.button_predict = tk.Button(self.frame, text="Предсказать", command=self.on_button_click)
        self.button_predict.grid(row=1, column=0, padx=5)

        self.button_delete = tk.Button(self.frame, text="Удалить последнее число", command=self.delete_last_number)
        self.button_delete.grid(row=1, column=1, padx=5)

        self.result_label = tk.Label(self.frame, text="")
        self.result_label.grid(row=2, columnspan=2)

        self.output_history_label = tk.Label(self.frame, text="Введенные числа:")
        self.output_history_label.grid(row=3, column=0, columnspan=2)

        self.output_history = tk.Listbox(self.frame)
        self.output_history.grid(row=4, column=0, columnspan=2)

        self.optimization_label = tk.Label(self.frame, text="Метод обучения:")
        self.optimization_label.grid(row=5, column=0, sticky=tk.E)

        self.optimization_combobox = ttk.Combobox(self.frame, textvariable=self.optimization_method)
        self.optimization_combobox['values'] = ('kNN', 'Случайный лес', 'Логистическая регрессия', 'Наивный байесовский')
        self.optimization_combobox.grid(row=5, column=1, sticky=tk.W)
        self.optimization_combobox.bind("<<ComboboxSelected>>", self.on_optimization_method_change)

        self.param_label = tk.Label(self.frame, text="Параметры:")
        self.param_label.grid(row=6, column=0, sticky=tk.E)

        self.params_frame = tk.Frame(self.frame)
        self.params_frame.grid(row=6, column=1, sticky=tk.W)

        params = {
            'n_neighbors': 'Число соседей',
            'n_estimators': 'Число деревьев',
            'max_depth': 'Макс. глубина деревьев',
            'C': 'Параметр регуляризации (C)'
        }
        self.param_entries = {}
        for param, label_text in params.items():
            label = tk.Label(self.params_frame, text=label_text)
            label.pack(anchor=tk.W)
            entry = tk.Entry(self.params_frame, textvariable=self.param_values[param])
            entry.pack(anchor=tk.W)
            self.param_entries[param] = entry

        self.load_button = tk.Button(self.frame, text="Загрузить историю", command=self.load_history)
        self.load_button.grid(row=7, column=0, padx=5)

        self.save_button = tk.Button(self.frame, text="Сохранить историю", command=self.save_history)
        self.save_button.grid(row=7, column=1, padx=5)

        self.prediction_count_label = tk.Label(self.frame, text="Число предсказаний:")
        self.prediction_count_label.grid(row=8, column=0, sticky=tk.E)

        self.prediction_count_entry = tk.Entry(self.frame)
        self.prediction_count_entry.insert(tk.END, "12")
        self.prediction_count_entry.grid(row=8, column=1, sticky=tk.W)

    def predict_next_numbers(self, num_predictions=12):
        if len(self.history) < 3:
            return "Не хватает данных для предсказания."

        model = self.get_model()

        param_grid = self.get_param_grid()

        X_train, y_train = self.prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        next_numbers = grid_search.predict_proba([[self.history[-3], self.history[-2], self.history[-1]]])[0]
        top_predictions = sorted(range(len(next_numbers)), key=lambda i: next_numbers[i], reverse=True)[:num_predictions]
        return sorted(top_predictions)

    def get_model(self):
        if self.optimization_method.get() == 'kNN':
            return KNeighborsClassifier()
        elif self.optimization_method.get() == 'Случайный лес':
            return RandomForestClassifier()
        elif self.optimization_method.get() == 'Логистическая регрессия':
            return LogisticRegression()
        elif self.optimization_method.get() == 'Наивный байесовский':
            return GaussianNB()

    def get_param_grid(self):
        param_grid = {}
        if self.optimization_method.get() in ['kNN', 'Случайный лес']:
            param_grid['n_neighbors'] = [int(self.param_values['n_neighbors'].get())]
        if self.optimization_method.get() == 'Случайный лес':
            param_grid['n_estimators'] = [int(self.param_values['n_estimators'].get())]
            param_grid['max_depth'] = [int(self.param_values['max_depth'].get())]
        if self.optimization_method.get() == 'Логистическая регрессия':
            param_grid['C'] = [float(self.param_values['C'].get())]
        return param_grid

    def prepare_data(self):
        X_train = [[self.history[i], self.history[i + 1], self.history[i + 2]] for i in range(len(self.history) - 3)]
        y_train = [self.history[i + 3] for i in range(len(self.history) - 3)]
        return X_train, y_train

    def save_history(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.history, f)

    def load_history(self):
        filename = filedialog.askopenfilename(filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
        if filename:
            with open(filename, 'rb') as f:
                self.history = pickle.load(f)
            self.update_output_history()

    def delete_last_number(self):
        if self.history:
            last_number = self.history.pop()
            self.update_output_history()
            self.button_delete.config(text=f"Удалить последнее число ({last_number})")

    def on_button_click(self, event=None):
        try:
            current_number = int(self.entry.get())
            if current_number == -1:
                self.save_history()
                self.root.destroy()
            elif current_number < 0 or current_number > 36:
                self.result_label.config(text="Пожалуйста, введите корректное число от 0 до 36.")
            else:
                self.history.append(current_number)
                next_numbers = self.predict_next_numbers(int(self.prediction_count_entry.get()))
                self.result_label.config(text=f"Предполагаемые следующие числа: {next_numbers}")
                self.update_output_history()
                self.entry.delete(0, tk.END)
                self.button_delete.config(text="Удалить последнее число")
        except ValueError:
            self.result_label.config(text="Пожалуйста, введите корректное число от 0 до 36.")

    def on_optimization_method_change(self, event=None):
        self.set_default_params()
        self.update_params_visibility()

    def set_default_params(self):
        method = self.optimization_method.get()
        if method in self.default_params:
            params = self.default_params[method]
            for param, value in params.items():
                self.param_values[param].set(value)

    def update_params_visibility(self):
        method = self.optimization_method.get()
        for param, entry in self.param_entries.items():
            if param in self.default_params[method]:
                entry.pack(anchor=tk.W)
            else:
                entry.pack_forget()

    def update_output_history(self):
        self.output_history.delete(0, tk.END)
        for number in self.history:
            self.output_history.insert(tk.END, number)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = RoulettePredictor()
    app.run()
