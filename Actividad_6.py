import tkinter as tk
from tkinter import messagebox, ttk
import heapq
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging
import os
from typing import Dict, Set, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoftwareProjectPlannerML:
    """Sistema experto para planificación de proyectos de desarrollo de software en un entorno de oficina real, con búsqueda UCS, ML dinámico (Red Neuronal) y selección individual de procesos y subprocesos."""
    def __init__(self, root: tk.Tk) -> None:
        """Inicializa la aplicación con interfaz gráfica, modelo ML (Red Neuronal), y selección de procesos y subprocesos."""
        self.root = root
        self.root.title("Sistema Experto de Planificación de Proyectos de Software")
        self.root.geometry("1400x800")
        self.root.resizable(False, False)
        # Estructuras de datos
        self.graph: Dict[str, List[Tuple[str, int]]] = {}
        self.states: Set[str] = set()
        self.model: Optional[nn.Module] = None
        self.model_trained: bool = False
        self.model_path: str = "software_project_planner_model_dynamic.pkl"
        self.training_data: List[Dict] = []
        # Mapa de códigos a nombres de procesos y subprocesos con dependencias
        self.process_map = {
            'A': {
                'name': 'Recopilación de Requerimientos',
                'depends_on': [],
                'subprocesses': {
                    'A1': 'Análisis de Necesidades',
                    'A2': 'Definición de Requerimientos',
                    'A3': 'Validación con Cliente'
                }
            },
            'B': {
                'name': 'Diseño de Arquitectura',
                'depends_on': ['A'],
                'subprocesses': {
                    'B1': 'Diseño de Base de Datos',
                    'B2': 'Diseño de Interfaces',
                    'B3': 'Modelado de Sistema'
                }
            },
            'C': {
                'name': 'Implementación del Código',
                'depends_on': ['A', 'B'],
                'subprocesses': {
                    'C1': 'Desarrollo Frontend',
                    'C2': 'Desarrollo Backend',
                    'C3': 'Integración de Módulos'
                }
            },
            'D': {
                'name': 'Pruebas y Testing',
                'depends_on': ['A', 'B', 'C'],
                'subprocesses': {
                    'D1': 'Pruebas Unitarias',
                    'D2': 'Pruebas de Integración',
                    'D3': 'Pruebas de Regresión'
                }
            },
            'E': {
                'name': 'Despliegue y Mantenimiento',
                'depends_on': ['A', 'B', 'C', 'D'],
                'subprocesses': {
                    'E1': 'Despliegue en Producción',
                    'E2': 'Soporte Técnico',
                    'E3': 'Actualizaciones'
                }
            }
        }
        # Configurar estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", padding=10, font=("Helvetica", 12, "bold"), background="#2196F3", foreground="white")
        self.style.configure("TLabel", font=("Helvetica", 12), foreground="#333")
        self.style.configure("TEntry", padding=8, font=("Helvetica", 12))
        self.style.configure("TCombobox", padding=8, font=("Helvetica", 12))
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("Treeview", font=("Helvetica", 11))
        self.style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))
        # Construir interfaz
        self._build_ui()
        self._update_graph_display()
        self._update_comboboxes()
        self._update_time()

    def _build_ui(self) -> None:
        """Construye una interfaz con pestañas, Treeview más ancho y validación en tiempo real."""
        main_frame = ttk.Frame(self.root, padding="20", style="TFrame")
        main_frame.pack(fill="both", expand=True)
        # Crear pestañas con ttk.Notebook
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, pady=10)
        # Pestaña 1: Gestión de Transiciones
        transition_frame = ttk.LabelFrame(notebook, text="Gestión de Transiciones", padding="15")
        notebook.add(transition_frame, text="Gestión de Transiciones")
        # Frame para Estado Actual y Estado Siguiente
        input_frame = ttk.Frame(transition_frame, padding="10")
        input_frame.pack(fill="x", pady=10)
        ttk.Label(input_frame, text="Estado Actual:", style="TLabel").grid(row=0, column=0, padx=15, pady=8, sticky="e")
        self.current_state_combo = ttk.Combobox(input_frame, width=50, values=["Inicio"], state="readonly", style="TCombobox")
        self.current_state_combo.grid(row=0, column=1, padx=15, pady=8, sticky="w")
        self.current_state_combo.set("Inicio")
        self.current_state_combo.bind("<<ComboboxSelected>>", self._update_comboboxes)
        ttk.Label(input_frame, text="Estado Siguiente:", style="TLabel").grid(row=0, column=2, padx=15, pady=8, sticky="e")
        self.next_state_tree = ttk.Treeview(input_frame, columns=("Proceso", "Subproceso"), show="headings", height=10)
        self.next_state_tree.heading("Proceso", text="Proceso")
        self.next_state_tree.column("Proceso", width=300)
        self.next_state_tree.heading("Subproceso", text="Subproceso")
        self.next_state_tree.column("Subproceso", width=300)
        self.next_state_tree.grid(row=0, column=3, padx=15, pady=8, sticky="w")
        self.next_state_tree.bind("<<TreeviewSelect>>", self._update_selected_state)
        ttk.Label(input_frame, text="Costo (horas):", style="TLabel").grid(row=1, column=0, padx=15, pady=8, sticky="e")
        self.cost_entry = ttk.Entry(input_frame, width=53, validate="key", style="TEntry")
        self.cost_entry.grid(row=1, column=1, columnspan=3, padx=15, pady=8, sticky="w")
        self.cost_entry.configure(validatecommand=(self.root.register(self._validate_cost), "%P"))
        # Botones de acción
        button_frame_trans = ttk.Frame(transition_frame)
        button_frame_trans.pack(fill="x", pady=15)
        ttk.Button(button_frame_trans, text="Agregar Transición", command=self._add_or_modify_transition, style="TButton").pack(side="left", padx=10)
        ttk.Button(button_frame_trans, text="Ayuda", command=self._show_help, style="TButton").pack(side="left", padx=10)
        ttk.Button(button_frame_trans, text="Restablecer", command=self._reset_to_default, style="TButton").pack(side="left", padx=10)
        ttk.Button(button_frame_trans, text="Ver Selecciones", command=self._show_selections, style="TButton").pack(side="left", padx=10)
        # Pestaña 2: Configuración de Búsqueda
        goal_frame = ttk.LabelFrame(notebook, text="Búsqueda y Predicción", padding="15")
        notebook.add(goal_frame, text="Búsqueda y Predicción")
        ttk.Label(goal_frame, text="Estado Inicial:", style="TLabel").grid(row=0, column=0, padx=15, pady=8, sticky="e")
        self.start_combo = ttk.Combobox(goal_frame, width=50, values=["Inicio"], state="readonly", style="TCombobox")
        self.start_combo.grid(row=0, column=1, padx=15, pady=8)
        self.start_combo.set("Inicio")
        ttk.Label(goal_frame, text="Estado Objetivo:", style="TLabel").grid(row=1, column=0, padx=15, pady=8, sticky="e")
        self.goal_combo = ttk.Combobox(goal_frame, width=50, values=["Inicio"], state="readonly", style="TCombobox")
        self.goal_combo.grid(row=1, column=1, padx=15, pady=8)
        button_frame_goal = ttk.Frame(goal_frame)
        button_frame_goal.grid(row=2, column=0, columnspan=2, pady=15)
        ttk.Button(button_frame_goal, text="Ejecutar Búsqueda UCS", command=self._run_ucs_search, style="TButton").grid(row=0, column=0, padx=10)
        ttk.Button(button_frame_goal, text="Predecir Costo", command=self._predict_cost, style="TButton").grid(row=0, column=1, padx=10)
        ttk.Button(button_frame_goal, text="Entrenar Modelo", command=self._train_model, style="TButton").grid(row=0, column=2, padx=10)
        # Pestaña 3: Resultados
        result_frame = ttk.LabelFrame(notebook, text="Resultados", padding="15")
        notebook.add(result_frame, text="Resultados")
        self.result_text = tk.Text(result_frame, height=20, width=120, font=("Courier New", 12), bg="#ffffff", fg="#333", wrap="word")
        self.result_text.pack(fill="both", expand=True)
        self.result_text.config(state="normal")
        # Etiqueta de hora
        self.time_label = ttk.Label(main_frame, text="", font=("Helvetica", 10), foreground="#666")
        self.time_label.pack(side="bottom", pady=5)
        # Variable para almacenar el estado seleccionado
        self.selected_state = ""

    def _validate_cost(self, value: str) -> bool:
        """Valida en tiempo real que el costo sea un número entero positivo."""
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False

    def _update_selected_state(self, event):
        """Actualiza el estado seleccionado basado en las selecciones del Treeview."""
        selected_items = [self.next_state_tree.item(item)["values"] for item in self.next_state_tree.selection()]
        states = []
        for item in selected_items:
            process = item[0].strip()
            subprocess = item[1].strip()
            if subprocess:
                states.append(f"{process} {subprocess}" if process else subprocess)
            elif process:
                states.append(process)
        self.selected_state = ", ".join(states) if states else ""
        logger.info(f"Estado seleccionado actualizado: {self.selected_state}")

    def _get_codes_from_desc(self, desc: str) -> Set[str]:
        """Convierte una descripción de procesos y subprocesos a códigos."""
        if desc == "Inicio":
            return set()
        tasks = [t.strip() for t in desc.split(',')]
        codes = set()
        for task in tasks:
            found = False
            for code, info in self.process_map.items():
                if info['name'] == task:
                    codes.add(code)
                    found = True
                    break
                for subcode, subname in info['subprocesses'].items():
                    full_name = f"{info['name']} {subname}"
                    if full_name == task or subname == task:
                        codes.add(subcode)
                        found = True
                        break
            if not found:
                logger.warning(f"Task '{task}' not found in process_map")
        return codes

    def _reset_to_default(self) -> None:
        """Restablece el grafo y datos a un estado vacío."""
        self.graph = {}
        self.states = set()
        self.training_data = []
        self._update_graph_display()
        self._update_comboboxes()
        messagebox.showinfo("Restablecido", "El grafo ha sido restablecido a un estado vacío.")

    def _show_help(self) -> None:
        """Muestra una ventana de ayuda con instrucciones detalladas."""
        help_text = (
            "Instrucciones para usar el Sistema Experto:\n\n"
            "Planifica proyectos de software con transiciones paso a paso y dependencias.\n"
            "Procesos y subprocesos disponibles:\n"
            "- Recopilación de Requerimientos (Análisis de Necesidades, Definición de Requerimientos, Validación con Cliente)\n"
            "- Diseño de Arquitectura (Diseño de Base de Datos, Diseño de Interfaces, Modelado de Sistema)\n"
            "- Implementación del Código (Desarrollo Frontend, Desarrollo Backend, Integración de Módulos)\n"
            "- Pruebas y Testing (Pruebas Unitarias, Pruebas de Integración, Pruebas de Regresión)\n"
            "- Despliegue y Mantenimiento (Despliegue en Producción, Soporte Técnico, Actualizaciones)\n\n"
            "1. **Estado Actual**: Selecciona el estado actual desde la lista.\n"
            "2. **Estado Siguiente**: Selecciona procesos/subprocesos en el Treeview.\n"
            "3. **Costo**: Ingresa horas (número entero positivo).\n"
            "4. **Estado Inicial y Objetivo**: Selecciona para búsqueda o predicción.\n"
            "5. **Botones**:\n"
            " - **Agregar Transición**: Añade/edita transiciones.\n"
            " - **Ayuda**: Muestra estas instrucciones.\n"
            " - **Restablecer**: Vuelve a un estado vacío.\n"
            " - **Ver Selecciones**: Muestra selecciones actuales.\n"
            " - **Ejecutar Búsqueda UCS**: Encuentra el camino óptimo.\n"
            " - **Predecir Costo**: Predice con la red neuronal.\n"
            " - **Entrenar Modelo**: Entrena con al menos 4 transiciones.\n\n"
            "Ejemplo:\n"
            "1. Inicio -> 'Análisis de Necesidades' (8h).\n"
            "2. 'Análisis de Necesidades' -> 'Definición de Requerimientos' (5h).\n"
            "3. 'Definición de Requerimientos' -> 'Validación con Cliente' (6h)."
        )
        messagebox.showinfo("Ayuda - Sistema Experto", help_text)

    def _update_comboboxes(self, event=None) -> None:
        """Actualiza las listas desplegables y el Treeview con procesos y subprocesos disponibles."""
        state_names = ["Inicio"]
        for state in sorted(list(self.states)):
            if state != "''":
                tasks = state.split(',')
                process_name = []
                for task in tasks:
                    if task in self.process_map:
                        process_name.append(self.process_map[task]['name'])
                    else:
                        for code, info in self.process_map.items():
                            if task in info['subprocesses']:
                                process_name.append(info['subprocesses'][task])
                state_names.append(", ".join(process_name))
        
        self.current_state_combo['values'] = state_names
        self.start_combo['values'] = state_names
        self.goal_combo['values'] = state_names
        if not self.current_state_combo.get() or self.current_state_combo.get() not in state_names:
            self.current_state_combo.set("Inicio")
        if not self.start_combo.get() or self.start_combo.get() not in state_names:
            self.start_combo.set("Inicio")
        if not self.goal_combo.get() or self.goal_combo.get() not in state_names:
            self.goal_combo.set(state_names[-1] if state_names != ["Inicio"] else "Inicio")
        
        # Actualizar Treeview con estructura jerárquica
        self.next_state_tree.delete(*self.next_state_tree.get_children())
        current_tasks = self._get_codes_from_desc(self.current_state_combo.get())
        
        # Mantener selecciones previas si son válidas
        previous_selection = self.selected_state.split(", ") if self.selected_state else []
        
        # Verificar qué procesos están completos
        completed_processes = set()
        for proc_code, proc_info in self.process_map.items():
            subproc_codes = set(proc_info['subprocesses'].keys())
            if subproc_codes.issubset(current_tasks) or proc_code in current_tasks:
                completed_processes.add(proc_code)
        
        # Identificar el proceso padre del estado actual
        current_process = None
        for task in current_tasks:
            if task in self.process_map:  # Es un proceso principal
                current_process = task
            else:  # Es un subproceso
                for proc_code, proc_info in self.process_map.items():
                    if task in proc_info['subprocesses']:
                        current_process = proc_code
                        break
        
        for code, info in self.process_map.items():
            # Validar si el proceso es válido (sus dependencias están completas)
            process_valid = all(dep in completed_processes for dep in info['depends_on'])
            is_completed = code in completed_processes
            
            # Mostrar el proceso si es válido, no está completado, y no es el estado actual
            parent_id = ""
            if process_valid and not is_completed and code not in current_tasks:
                parent_id = self.next_state_tree.insert("", "end", values=(info['name'], ""), text=info['name'])
            
            # Mostrar subprocesos si el proceso es válido o es el proceso actual
            if (process_valid and not is_completed) or (code == current_process):
                for subcode, subname in info['subprocesses'].items():
                    if subcode not in current_tasks:
                        sub_id = self.next_state_tree.insert(parent_id if parent_id else "", "end", values=("", subname), text=subname)
                        full_name = f"{info['name']} {subname}"
                        # Restaurar selección previa si es válida
                        if full_name in previous_selection or subname in previous_selection:
                            self.next_state_tree.selection_add(sub_id)
        
        self._update_selected_state(None)

    def _add_or_modify_transition(self) -> None:
        """Agrega o modifica una transición en el grafo y almacena datos para ML, validando dependencias."""
        logger.info("Iniciando _add_or_modify_transition")
        current_state_desc = self.current_state_combo.get()
        next_state_desc = self.selected_state
        cost_str = self.cost_entry.get().strip()
        logger.info(f"Valores recibidos - current_state_desc: {current_state_desc}, next_state_desc: {next_state_desc}, cost_str: {cost_str}")
        
        try:
            if not current_state_desc or not next_state_desc or not cost_str:
                raise ValueError("Todos los campos deben estar completos. Asegúrate de seleccionar un proceso/subproceso en el Treeview y un costo válido.")
            
            cost = int(cost_str)
            if cost <= 0:
                raise ValueError("El costo debe ser un número entero positivo.")
            # Convertir current_state_desc a código
            current_state = "''" if current_state_desc == "Inicio" else ','.join(self._get_codes_from_desc(current_state_desc))
            # Convertir next_state_desc a código
            next_state_tasks = [t.strip() for t in next_state_desc.split(',')] if next_state_desc else []
            next_state_codes = set()
            for task in next_state_tasks:
                found = False
                for code, info in self.process_map.items():
                    if info['name'] == task or f"{info['name']} {task}" == task:
                        next_state_codes.add(code)
                        found = True
                        break
                    for subcode, subname in info['subprocesses'].items():
                        full_name = f"{info['name']} {subname}"
                        if full_name == task or subname == task:
                            next_state_codes.add(subcode)
                            found = True
                            break
                if not found:
                    raise ValueError(f"El proceso o subproceso '{task}' no es válido o no está disponible.")
            next_state = ','.join(sorted(next_state_codes))
            if not self._is_valid_state(next_state):
                raise ValueError("El estado siguiente debe contener solo procesos o subprocesos válidos separados por comas.")
            logger.info(f"Convertido - current_state: {current_state}, next_state: {next_state}")
            # Validar dependencias
            current_tasks = set(current_state.split(',')) if current_state != "''" else set()
            next_tasks = set(next_state.split(','))
            completed_processes = set()
            for proc_code, proc_info in self.process_map.items():
                if set(proc_info['subprocesses'].keys()).issubset(current_tasks) or proc_code in current_tasks:
                    completed_processes.add(proc_code)
            for task in next_tasks - current_tasks:
                if task in self.process_map:
                    if not all(dep in completed_processes for dep in self.process_map[task]['depends_on']):
                        raise ValueError(f"{self.process_map[task]['name']} requiere que se completen: {', '.join(self.process_map[dep]['name'] for dep in self.process_map[task]['depends_on'])} primero.")
                else:
                    for code, info in self.process_map.items():
                        if task in info['subprocesses']:
                            if not all(dep in completed_processes for dep in info['depends_on']):
                                raise ValueError(f"{info['subprocesses'][task]} requiere que se completen: {', '.join(self.process_map[d]['name'] for d in info['depends_on'])} primero.")
            self.states.add(current_state)
            self.states.add(next_state)
            if current_state not in self.graph:
                self.graph[current_state] = []
            for i, (state, _) in enumerate(self.graph[current_state]):
                if state == next_state:
                    self.graph[current_state][i] = (next_state, cost)
                    break
            else:
                self.graph[current_state].append((next_state, cost))
            num_tareas = len(next_state.split(',')) if next_state and next_state != "''" else 0
            dependencias = sum(1 for s in self.graph if s != "''" and any(t in next_state for t in s.split(',')))
            self.training_data.append({
                'num_tareas': num_tareas,
                'dependencias': dependencias,
                'duracion_promedio': cost,
                'costo_total': cost
            })
            logger.info("Transición agregada/modificada: %s -> %s, costo: %d", current_state_desc, next_state_desc, cost)
            messagebox.showinfo("Éxito", f"Transición de {current_state_desc} a {next_state_desc} con costo {cost} horas agregada.")
            self.cost_entry.delete(0, tk.END)
            
            # Mantener la selección en el Treeview si es válida
            self._update_comboboxes()
            if next_state_desc in [f"{info['name']} {subname}" for code, info in self.process_map.items() for subname in info['subprocesses'].values()] or next_state_desc in [info['name'] for info in self.process_map.values()]:
                for item in self.next_state_tree.get_children():
                    values = self.next_state_tree.item(item)["values"]
                    full_name = f"{values[0]} {values[1]}".strip() if values[1] else values[0]
                    if full_name == next_state_desc:
                        self.next_state_tree.selection_add(item)
                        break
            self._update_selected_state(None)
            self._update_graph_display()

        except ValueError as e:
            logger.error("Error en transición: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}. Revisa los logs para más detalles.")

    def _is_valid_state(self, state: str) -> bool:
        """Valida que un estado contenga solo códigos válidos de procesos o subprocesos."""
        if state == "''":
            return True
        tasks = state.split(',')
        valid_codes = set(self.process_map.keys())
        for info in self.process_map.values():
            valid_codes.update(info['subprocesses'].keys())
        return all(task in valid_codes for task in tasks)

    def _train_model(self) -> None:
        """Entrena una Red Neuronal simple con PyTorch para aprendizaje automático."""
        try:
            if not self.training_data:
                raise ValueError("No hay datos registrados para entrenar el modelo.")
            df = pd.DataFrame(self.training_data)
            X = df[['num_tareas', 'dependencias', 'duracion_promedio']].values
            y = df['costo_total'].values.reshape(-1, 1)
            if len(df) < 4:
                raise ValueError("Se necesitan al menos 4 transiciones registradas para entrenar el modelo.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            class NeuralNet(nn.Module):
                def __init__(self, input_size):
                    super(NeuralNet, self).__init__()
                    self.fc1 = nn.Linear(input_size, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 1)
                    self.relu = nn.ReLU()
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            self.model = NeuralNet(input_size=3)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            epochs = 200
            for epoch in range(epochs):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 50 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            self.model.eval()
            with torch.no_grad():
                y_pred_tensor = self.model(X_test_tensor)
                y_pred = y_pred_tensor.numpy()
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            torch.save(self.model.state_dict(), self.model_path)
            self.model_trained = True
            logger.info("Red Neuronal entrenada con %d registros. MSE: %.2f, R2: %.2f", len(df), mse, r2)
            self._update_graph_display()
            self.result_text.insert(tk.END, f"\nRed Neuronal entrenada con {len(df)} registros. MSE: {mse:.2f}, R2: {r2:.2f}\n")
            self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
            self.result_text.update_idletasks()
        except ValueError as e:
            logger.error("Error al entrenar la Red Neuronal: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado al entrenar la Red Neuronal: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def _update_graph_display(self) -> None:
        """Actualiza la visualización del grafo en la interfaz con nombres de procesos y subprocesos."""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Estructura del Proyecto (Transiciones Paso a Paso):\n\n")
        
        for state, transitions in sorted(self.graph.items()):
            if state == "''":
                state_desc = "Inicio"
            else:
                tasks = state.split(',')
                process_name = []
                for task in tasks:
                    if task in self.process_map:
                        process_name.append(self.process_map[task]['name'])
                    else:
                        for code, info in self.process_map.items():
                            if task in info['subprocesses']:
                                process_name.append(info['subprocesses'][task])
                state_desc = ", ".join(process_name)
            
            self.result_text.insert(tk.END, f"{state_desc}:\n")
            for next_state, cost in transitions:
                if next_state == "''":
                    next_state_desc = "Inicio"
                else:
                    next_tasks = next_state.split(',')
                    next_process_name = []
                    for task in next_tasks:
                        if task in self.process_map:
                            next_process_name.append(self.process_map[task]['name'])
                        else:
                            for code, info in self.process_map.items():
                                if task in info['subprocesses']:
                                    next_process_name.append(info['subprocesses'][task])
                    next_state_desc = ", ".join(next_process_name)
                
                self.result_text.insert(tk.END, f" -> {next_state_desc}: {cost} horas\n")
            self.result_text.insert(tk.END, "\n")
        
        self.result_text.insert(tk.END, f"Estados registrados: {sorted(list(self.states))}\n")
        if self.training_data:
            self.result_text.insert(tk.END, f"\nDatos para ML: {len(self.training_data)} transiciones registradas\n")
        self.result_text.config(state="normal")
        self.result_text.update_idletasks()

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
        """Ejecuta la búsqueda UCS y muestra los resultados con nombres de procesos y subprocesos."""
        start_state_desc = self.start_combo.get()
        goal_state_desc = self.goal_combo.get()
        try:
            if not start_state_desc or not goal_state_desc:
                raise ValueError("Debe seleccionar un estado inicial y un estado objetivo.")
            
            # Convertir descripciones a códigos
            start_state = "''" if start_state_desc == "Inicio" else ','.join(self._get_codes_from_desc(start_state_desc))
            goal_state = "''" if goal_state_desc == "Inicio" else ','.join(self._get_codes_from_desc(goal_state_desc))
            if start_state not in self.states or goal_state not in self.states:
                raise ValueError("El estado inicial o objetivo no está definido en el grafo.")
            path, total_cost = self._uniform_cost_search(start_state, goal_state)
            self._update_graph_display()
            if path:
                path_desc = []
                for state in path:
                    if state == "''":
                        path_desc.append("Inicio")
                    else:
                        tasks = state.split(',')
                        state_desc = []
                        for task in tasks:
                            if task in self.process_map:
                                state_desc.append(self.process_map[task]['name'])
                            else:
                                for code, info in self.process_map.items():
                                    if task in info['subprocesses']:
                                        state_desc.append(info['subprocesses'][task])
                        path_desc.append(", ".join(state_desc))
                
                self.result_text.insert(tk.END, f"\nCamino óptimo paso a paso:\n")
                for i, desc in enumerate(path_desc, 1):
                    self.result_text.insert(tk.END, f" {i}. {desc}\n")
                self.result_text.insert(tk.END, f"\nCosto total estimado: {total_cost} horas\n")
                self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
                self.result_text.update_idletasks()
                logger.info("Búsqueda UCS completada: Camino=%s, Costo=%d", path_desc, total_cost)
            else:
                self.result_text.insert(tk.END, "\nNo se encontró solución. Verifica las dependencias.\n")
                self.result_text.update_idletasks()
                logger.warning("No se encontró solución para la búsqueda UCS")
        except ValueError as e:
            logger.error("Error en búsqueda UCS: %s", str(e))
            messagebox.showerror("Error", str(e))

    def _predict_cost(self) -> None:
        """Predice el costo total usando la Red Neuronal entrenada."""
        if not self.model_trained:
            messagebox.showerror("Error", "Entrena la Red Neuronal primero.")
            return
        start_state_desc = self.start_combo.get().strip()
        goal_state_desc = self.goal_combo.get().strip()
        try:
            if not start_state_desc or not goal_state_desc:
                raise ValueError("Debe seleccionar un estado inicial y un estado objetivo.")
            # Convertir a códigos
            goal_state = "''" if goal_state_desc == "Inicio" else ','.join(self._get_codes_from_desc(goal_state_desc))
            num_tareas = len(goal_state.split(',')) if goal_state != "''" else 1
            dependencias = sum(1 for state in self.graph if state != "''" and any(t in goal_state for t in state.split(',')))
            duracion_promedio = np.mean([cost for state in self.graph for _, cost in self.graph[state]]) if self.graph else 1.0
            features = np.array([[num_tareas, dependencias, duracion_promedio]])
            features_tensor = torch.tensor(features, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                predicted_cost = self.model(features_tensor).item()
            self._update_graph_display()
            self.result_text.insert(tk.END, f"\nPredicción de costo para {goal_state_desc}: {predicted_cost:.2f} horas\n")
            self.result_text.insert(tk.END, f"Características: Tareas={num_tareas}, Dependencias={dependencias}, Dur. Promedio={duracion_promedio:.2f}\n")
            self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
            self.result_text.update_idletasks()
            logger.info("Predicción Red Neuronal: Costo=%.2f, Características=%s", predicted_cost, features)
        except ValueError as e:
            logger.error("Error en predicción Red Neuronal: %s", str(e))
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logger.error("Error inesperado en predicción Red Neuronal: %s", str(e))
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def _show_selections(self) -> None:
        """Muestra las selecciones actuales del Treeview."""
        selected_items = [self.next_state_tree.item(item)["values"] for item in self.next_state_tree.selection()]
        self.result_text.insert(tk.END, f"\nSelecciones actuales: {', '.join([f'{item[0]} {item[1]}'.strip() if item[1] else item[0] for item in selected_items if item[0] or item[1]]) if selected_items else 'Ninguna'}\n")
        self.result_text.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}\n")
        self.result_text.update_idletasks()
        logger.info("Selecciones mostradas: %s", selected_items)

    def _update_time(self):
        """Actualiza la hora en la interfaz."""
        current_time = datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')
        self.time_label.config(text=f"Hora actual: {current_time}")
        self.root.after(60000, self._update_time)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SoftwareProjectPlannerML(root)
        root.mainloop()
    except Exception as e:
        logger.critical("Error al iniciar la aplicación: %s", str(e))
        messagebox.showerror("Error Crítico", f"No se pudo iniciar la aplicación: {str(e)}")