import tkinter as tk
from tkinter import messagebox, ttk
import heapq
from datetime import datetime

class StatePlanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Planificación de Estados")
        self.root.geometry("700x500")

        # Diccionario para almacenar el grafo
        self.graph = {}
        # Conjunto para rastrear estados existentes
        self.states = set()

        # Frame para la entrada de transiciones
        self.transition_frame = ttk.LabelFrame(root, text="Definir Transiciones", padding="10")
        self.transition_frame.pack(fill="x", padx=10, pady=5)

        # Campos de entrada
        ttk.Label(self.transition_frame, text="Estado Actual (ej. '' para inicio):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.current_state_entry = ttk.Entry(self.transition_frame)
        self.current_state_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.transition_frame, text="Estado Siguiente (ej. 'A,B'):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.next_state_entry = ttk.Entry(self.transition_frame)
        self.next_state_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.transition_frame, text="Costo (entero positivo):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.cost_entry = ttk.Entry(self.transition_frame)
        self.cost_entry.grid(row=2, column=1, padx=5, pady=5)

        # Botones para agregar y modificar
        ttk.Button(self.transition_frame, text="Agregar/Modificar Transición", command=self.add_or_modify_transition).grid(row=3, column=0, columnspan=2, pady=10)

        # Frame para estados inicial y objetivo
        self.goal_frame = ttk.LabelFrame(root, text="Configurar Búsqueda", padding="10")
        self.goal_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self.goal_frame, text="Estado Inicial (ej. ''):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.start_entry = ttk.Entry(self.goal_frame)
        self.start_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.goal_frame, text="Estado Objetivo (ej. 'A,B,C,D,E'):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.goal_entry = ttk.Entry(self.goal_frame)
        self.goal_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.goal_frame, text="Ejecutar Búsqueda", command=self.run_search).grid(row=2, column=0, columnspan=2, pady=10)

        # Área de resultados
        self.result_frame = ttk.LabelFrame(root, text="Resultados", padding="10")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.result_text = tk.Text(self.result_frame, height=10, width=80)
        self.result_text.pack(fill="both", expand=True)

        # Mostrar grafo actual
        self.update_graph_display()

    def add_or_modify_transition(self):
        """Agrega o modifica una transición en el grafo con validaciones avanzadas."""
        current_state = self.current_state_entry.get().strip()
        next_state = self.next_state_entry.get().strip()
        try:
            cost = int(self.cost_entry.get().strip())
            if not current_state or not next_state or cost < 0:
                messagebox.showerror("Error", "Todos los campos deben estar completos y el costo debe ser positivo.")
                return

            # Validar que los estados sean únicos y consistentes
            if current_state not in self.states:
                self.states.add(current_state)
            if next_state not in self.states:
                self.states.add(next_state)

            # Inicializar lista de transiciones si no existe
            if current_state not in self.graph:
                self.graph[current_state] = []

            # Verificar si la transición ya existe para modificarla o agregarla
            transition_exists = False
            for i, (state, c) in enumerate(self.graph[current_state]):
                if state == next_state:
                    self.graph[current_state][i] = (next_state, cost)
                    transition_exists = True
                    break
            if not transition_exists:
                self.graph[current_state].append((next_state, cost))

            messagebox.showinfo("Éxito", f"Transición de {current_state} a {next_state} actualizada con costo {cost}.")
            self.current_state_entry.delete(0, tk.END)
            self.next_state_entry.delete(0, tk.END)
            self.cost_entry.delete(0, tk.END)
            self.update_graph_display()
        except ValueError:
            messagebox.showerror("Error", "El costo debe ser un número entero válido.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def update_graph_display(self):
        """Actualiza la visualización del grafo en el área de resultados."""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Estructura del Grafo:\n")
        for state, transitions in self.graph.items():
            self.result_text.insert(tk.END, f"{state}: {transitions}\n")
        self.result_text.insert(tk.END, f"\nEstados registrados: {sorted(list(self.states))}\n")

    def uniform_cost_search(self, start, goal):
        """
        Implementa la Búsqueda por Costo Uniforme (UCS) para encontrar el camino óptimo.
        Args:
            start (str): Estado inicial.
            goal (str): Estado objetivo.
        Returns:
            tuple: (camino, costo_total) o (None, inf) si no hay solución.
        """
        queue = [(0, start, [])]
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

    def run_search(self):
        """Ejecuta la búsqueda y muestra los resultados con validaciones."""
        start_state = self.start_entry.get().strip()
        goal_state = self.goal_entry.get().strip()
        
        if not start_state or not goal_state:
            messagebox.showerror("Error", "Debe ingresar un estado inicial y un estado objetivo.")
            return
        if start_state not in self.states or goal_state not in self.states:
            messagebox.showerror("Error", "El estado inicial o objetivo no está definido en el grafo.")
            return
        if start_state not in self.graph and start_state != '':
            messagebox.showerror("Error", "El estado inicial no tiene transiciones definidas.")
            return

        path, total_cost = self.uniform_cost_search(start_state, goal_state)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Estructura del Grafo:\n")
        for state, transitions in self.graph.items():
            self.result_text.insert(tk.END, f"{state}: {transitions}\n")
        self.result_text.insert(tk.END, f"\nEstados registrados: {sorted(list(self.states))}\n")
        if path:
            self.result_text.insert(tk.END, f"\nCamino óptimo: {path}\n")
            self.result_text.insert(tk.END, f"Costo total: {total_cost} unidades\n")
            self.result_text.insert(tk.END, f"Fecha y hora de ejecución: {datetime.now().strftime('%I:%M %p -%H, %A, %d de %B de %Y')}")
        else:
            self.result_text.insert(tk.END, "\nNo se encontró solución\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = StatePlanner(root)
    root.mainloop()