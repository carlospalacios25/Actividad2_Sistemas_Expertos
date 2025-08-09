import tkinter as tk
from tkinter import messagebox, ttk
import heapq
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging
import os
from typing import Dict, Set, List, Tuple, Optional

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventPlannerML:
    """Sistema experto para planificación de eventos con búsqueda UCS, ML dinámico y listas desplegables."""

    def __init__(self, root: tk.Tk) -> None:
        """Inicializa la aplicación con interfaz gráfica, modelo ML y listas desplegables."""
        self.root = root
        self.root.title("Sistema Experto de Planificación de Eventos")
        self.root.geometry("1000x800")
        self.root.resizable(False, False)

        # Estructuras de datos
        self.graph: Dict[str, List[Tuple[str, int]]] = {}
        self.states: Set[str] = set()
        self.model: Optional[RandomForestRegressor] = None
        self.model_trained: bool = False
        self.model_path: str = "event_planner_model_dynamic.pkl"
        self.training_data: List[Dict] = []

        # Configurar estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", padding=8, font=("Helvetica", 11, "bold"), background="#4CAF50", foreground="white")
        self.style.configure("TLabel", font=("Helvetica", 11), foreground="#333")
        self.style.configure("TEntry", padding=6, font=("Helvetica", 11))
        self.style.configure("TCombobox", padding=6, font=("Helvetica", 11))
        self.style.configure("TFrame", background="#f0f0f0")

        # Construir interfaz
        self._build_ui()
        self._update_graph_display()

    def _build_ui(self) -> None:
        """Construye la interfaz gráfica con listas desplegables y secciones organizadas."""
        main_frame = ttk.Frame(self.root, padding="15", style="TFrame")
        main_frame.pack(fill="both", expand=True)

        # Frame para transiciones
        self.transition_frame = ttk.LabelFrame(main_frame, text="Gestión de Transiciones", padding="15")
        self.transition_frame.pack(fill="x", pady=10)

        ttk.Label(self.transition_frame, text="Estado Actual:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.current_state_combo = ttk.Combobox(self.transition_frame, width=32, values=["''"], state="normal")
        self.current_state_combo.grid(row=0, column=1, padx=10, pady=5)
        self.current_state_combo.set("''")

        ttk.Label(self.transition_frame, text="Estado Siguiente:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.next_state_entry = ttk.Entry(self.transition_frame, width=35)
        self.next_state_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.transition_frame, text="Costo (horas, entero):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.cost_entry = ttk.Entry(self.transition_frame, width=35)
        self.cost_entry.grid(row=2, column=1, padx=10, pady=5)

        button_frame_trans = ttk.Frame(self.transition_frame)
        button_frame_trans.grid(row=3, column=0, columnspan=2, pady=15)
        ttk.Button(button_frame_trans, text="Agregar Transición", command=self._add_or_modify_transition).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame_trans, text="Ayuda", command=self._show_help).grid(row=0, column=1, padx=5)

        # Frame para búsqueda
        self.goal_frame = ttk.LabelFrame(main_frame, text="Configuración de Búsqueda", padding="15")
        self.goal_frame.pack(fill="x", pady=10)

        ttk.Label(self.goal_frame, text="Estado Inicial:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.start_combo = ttk.Combobox(self.goal_frame, width=32, values=["''"], state="readonly")
        self.start_combo.grid(row=0, column=1, padx=10, pady=5)
        self.start_combo.set("''")

        ttk.Label(self.goal_frame, text="Estado Objetivo:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.goal_combo = ttk.Combobox(self.goal_frame, width=32, values=["''"], state="readonly")
        self.goal_combo.grid(row=1, column=1, padx=10, pady=5)

        button_frame_goal = ttk.Frame(self.goal_frame)
        button_frame_goal.grid(row=2, column=0, columnspan=2, pady=15)
        ttk.Button(button_frame_goal, text="Ejecutar Búsqueda UCS", command=self._run_ucs_search).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame_goal, text="Predecir Costo con ML", command=self._predict_cost).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame_goal, text="Entrenar Modelo ML", command=self._train_model).grid(row=0, column=2, padx=5)

        # Área de resultados
        self.result_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="15")
        self.result_frame.pack(fill="both", expand=True, pady=10)
        self.result_text = tk.Text(self.result_frame, height=20, width=100, font=("Courier New", 11), bg="#ffffff", fg="#333")
        self.result_text.pack(fill="both", expand=True)
        self.result_text.config(state="normal")

    def _show_help(self) -> None:
        """Muestra una ventana de ayuda con instrucciones detalladas."""
        help_text = (
            "Instrucciones para usar el Sistema Experto:\n\n"
            "1. **Estado Actual**: Selecciona el estado inicial de la transición desde la lista desplegable (ej. '' para inicio, 'A' para una tarea).\n"
            "   - La lista muestra los estados ya registrados.\n"
            "   - '' representa el estado inicial vacío.\n\n"
            "2. **Estado Siguiente**: Ingresa el estado al que se transita (ej. 'A' o 'A,B,C').\n"
            "   - Usa letras (A, B, C, etc.) separadas por comas para estados combinados.\n"
            "   - Asegúrate de respetar las dependencias del proyecto.\n\n"
            "3. **Costo**: Ingresa el costo en horas (número entero positivo) para la transición.\n\n"
            "4. **Estado Inicial y Objetivo**: Selecciona los estados inicial y objetivo desde las listas desplegables.\n"
            "   - Solo se muestran los estados que has registrado.\n\n"
            "5. **Botones**:\n"
            "   - **Agregar Transición**: Añade o modifica una transición en el grafo.\n"
            "   - **Ejecutar Búsqueda UCS**: Encuentra el camino óptimo usando Búsqueda por Costo Uniforme.\n"
            "   - **Predecir Costo con ML**: Predice el costo total usando el modelo de Machine Learning.\n"
            "   - **Entrenar Modelo ML**: Entrena el modelo con las transiciones registradas (mínimo 4).\n\n"
            "Ejemplo:\n"
            "Para un evento con tareas A (4h), B (5h, depende de A), C (3h, depende de A), D (6h, depende de A), E (2h, depende de B,C,D):\n"
            "1. Agrega transiciones: '' -> 'A' (4), 'A' -> 'A,B' (5), 'A' -> 'A,C' (3), 'A' -> 'A,D' (6), 'A,B,C,D' -> 'A,B,C,D,E' (2).\n"
            "2. Selecciona en las listas: Inicial = '', Objetivo = 'A,B,C,D,E'.\n"
            "3. Usa los botones para obtener resultados o predicciones."
        )
        messagebox.showinfo("Ayuda - Sistema Experto", help_text)

    def _update_comboboxes(self) -> None:
        """Actualiza las listas desplegables con los estados registrados."""
        states = sorted(list(self.states)) or ["''"]
        self.current_state_combo['values'] = states
        self.start_combo['values'] = states
        self.goal_combo['values'] = states
        if not self.current_state_combo.get() or self.current_state_combo.get() not in states:
            self.current_state_combo.set("''")
        if not self.start_combo.get() or self.start_combo.get() not in states:
            self.start_combo.set("''")
        if not self.goal_combo.get() or self.goal_combo.get() not in states:
            self.goal_combo.set(states[-1] if states != ["''"] else "''")

    def _add_or_modify_transition(self) -> None:
        """Agrega o modifica una transición en el grafo y almacena datos para ML."""
        current_state = self.current_state_combo.get().strip()
        next_state = self.next_state_entry.get().strip()
        cost_str = self.cost_entry.get().strip()

        try:
            if not current_state or not next_state or not cost_str:
                raise ValueError("Todos los campos deben estar completos.")
            
            cost = int(cost_str)
            if cost <= 0:
                raise ValueError("El costo debe ser un número entero positivo.")

            if not self._is_valid_state(next_state):
                raise ValueError("El estado siguiente debe contener solo letras, comas o ser ''.")

            # Agregar estados al conjunto
            self.states.add(current_state)
            self.states.add(next_state)

            # Inicializar transiciones si no existen
            if current_state not in self.graph:
                self.graph[current_state] = []

            # Actualizar o agregar transición
            for i, (state, _) in enumerate(self.graph[current_state]):
                if state == next_state:
                    self.graph[current_state][i] = (next_state, cost)
                    break
            else:
                self.graph[current_state].append((next_state, cost))

            # Almacenar datos para entrenamiento
            num_tareas = len(next_state.split(',')) if next_state and next_state != "''" else 0
            dependencias = sum(1 for s in self.graph if s != "''" and any(t in next_state for t in s.split(',')))
            self.training_data.append({
                'num_tareas': num_tareas,
                'dependencias': dependencias,
                'duracion_promedio': cost,
                'costo_total': cost
            })

            logger.info("Transición agregada/modificada: %s -> %s, costo: %d", current_state, next_state, cost)
            messagebox.showinfo("Éxito", f"Transición de {current_state} a {next_state} con costo {cost} horas.")
            self.next_state_entry.delete(0, tk.END)
            self.cost_entry.delete(0, tk.END)
            self._update_graph_display()
            self._update_comboboxes()
        except ValueError as e:
            logger.error("Error en transición: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def _is_valid_state(self, state: str) -> bool:
        """Valida que un estado sea válido (vacío o letras separadas por comas)."""
        if state == "''":
            return True
        return all(c.isalpha() or c == ',' for c in state)

    def _train_model(self) -> None:
        """Entrena el modelo RandomForest con los datos registrados por el usuario."""
        try:
            if not self.training_data:
                raise ValueError("No hay datos registrados para entrenar el modelo.")

            df = pd.DataFrame(self.training_data)
            X = df[['num_tareas', 'dependencias', 'duracion_promedio']]
            y = df['costo_total']

            if len(df) < 4:
                raise ValueError("Se necesitan al menos 4 transiciones registradas para entrenar el modelo.")

            # Dividir datos (80% entrenamiento, 20% prueba)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Entrenar modelo
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            self.model.fit(X_train, y_train)

            # Validar modelo
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Guardar modelo
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            self.model_trained = True
            logger.info("Modelo entrenado con %d registros. MSE: %.2f, R2: %.2f", len(df), mse, r2)
            self._update_graph_display()
            self.result_text.insert(tk.END, f"\nModelo entrenado con {len(df)} registros. MSE: {mse:.2f}, R2: {r2:.2f}\n")
            self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
        except ValueError as e:
            logger.error("Error al entrenar el modelo: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado al entrenar el modelo: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def _update_graph_display(self) -> None:
        """Actualiza la visualización del grafo en la interfaz."""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Estructura del Grafo:\n")
        for state, transitions in sorted(self.graph.items()):
            self.result_text.insert(tk.END, f"{state}: {transitions}\n")
        self.result_text.insert(tk.END, f"\nEstados registrados: {sorted(list(self.states))}\n")
        if self.training_data:
            self.result_text.insert(tk.END, f"\nDatos para ML: {len(self.training_data)} transiciones registradas\n")

    def _uniform_cost_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], float]:
        """Implementa la búsqueda por costo uniforme (UCS)."""
        queue = [(0, start, [start])]
        visited = set()

        while queue:
            cost, current, path = heapq.heappop(queue)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path, cost

            if current in self.graph:
                for next_state, task_cost in self.graph[current]:
                    if next_state not in visited:
                        new_cost = cost + task_cost
                        new_path = path + [next_state]
                        heapq.heappush(queue, (new_cost, next_state, new_path))

        return None, float('inf')

    def _run_ucs_search(self) -> None:
        """Ejecuta la búsqueda UCS y muestra los resultados."""
        start_state = self.start_combo.get().strip()
        goal_state = self.goal_combo.get().strip()

        try:
            if not start_state or not goal_state:
                raise ValueError("Debe seleccionar un estado inicial y un estado objetivo.")
            if start_state not in self.states or goal_state not in self.states:
                raise ValueError("El estado inicial o objetivo no está definido en el grafo.")
            if start_state not in self.graph and start_state != "''":
                raise ValueError("El estado inicial no tiene transiciones definidas.")

            path, total_cost = self._uniform_cost_search(start_state, goal_state)
            self._update_graph_display()

            if path:
                self.result_text.insert(tk.END, f"\nCamino óptimo: {path}\n")
                self.result_text.insert(tk.END, f"Costo total: {total_cost} horas\n")
                self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
                logger.info("Búsqueda UCS completada: Camino=%s, Costo=%d", path, total_cost)
            else:
                self.result_text.insert(tk.END, "\nNo se encontró solución\n")
                logger.warning("No se encontró solución para la búsqueda UCS")
        except ValueError as e:
            logger.error("Error en búsqueda UCS: %s", str(e))
            messagebox.showerror("Error", str(e))

    def _predict_cost(self) -> None:
        """Predice el costo total usando el modelo ML."""
        if not self.model_trained:
            messagebox.showerror("Error", "Entrena el modelo primero usando 'Entrenar Modelo ML'.")
            return

        start_state = self.start_combo.get().strip()
        goal_state = self.goal_combo.get().strip()

        try:
            if not start_state or not goal_state:
                raise ValueError("Debe seleccionar un estado inicial y un estado objetivo.")

            # Extraer características
            num_tareas = len(goal_state.split(',')) if goal_state and goal_state != "''" else 1
            dependencias = sum(1 for state in self.graph if state != "''" and any(t in goal_state for t in state.split(',')))
            duracion_promedio = np.mean([cost for state in self.graph for _, cost in self.graph[state]]) if self.graph else 1.0

            # Predecir costo
            features = np.array([[num_tareas, dependencias, duracion_promedio]])
            predicted_cost = self.model.predict(features)[0]

            self._update_graph_display()
            self.result_text.insert(tk.END, f"\nPredicción de costo con ML para {goal_state}: {predicted_cost:.2f} horas\n")
            self.result_text.insert(tk.END, f"Características: Tareas={num_tareas}, Dependencias={dependencias}, Duración promedio={duracion_promedio:.2f}\n")
            self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
            logger.info("Predicción ML: Costo=%.2f, Características=%s", predicted_cost, features)
        except ValueError as e:
            logger.error("Error en predicción ML: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado en predicción ML: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EventPlannerML(root)
        root.mainloop()
    except Exception as e:
        logger.critical("Error al iniciar la aplicación: %s", str(e))
        messagebox.showerror("Error Crítico", f"No se pudo iniciar la aplicación: {str(e)}")