from __future__ import annotations

import sys
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog, ttk
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.domain.models.world.entity_blueprint import EntityBlueprint
from src.domain.models.world.position import Position
from src.domain.models.world.world_definition import (
    PlacedAgentData,
    PlacedEntityData,
    WorldDefinition,
)
from src.infrastructure.repositories.json_blueprint_repository import JsonBlueprintRepository
from src.infrastructure.repositories.json_world_repository import JsonWorldRepository


class WorldEditorApp:
    CELL_SIZE = 18

    CATEGORY_NAMES = {
        "obstacle": "1. Hindernisse & Gelände",
        "entity": "2. Interaktive Objekte",
        "agent": "3. Agenten & Charaktere",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SocietySim - World Editor")
        self.root.geometry("1450x920")

        self.width = 90
        self.height = 45

        data_dir = project_root / "data"
        self.blueprint_repo = JsonBlueprintRepository(data_dir / "blueprints.json")
        self.world_repo = JsonWorldRepository(data_dir / "worlds")

        self.blueprints = self.blueprint_repo.get_all()

        # Editor-Zustand
        self.active_tool: str = "brush"  # "brush", "eraser", "picker"
        self.active_blueprint_id: str = self.blueprints[0].id if self.blueprints else "wall"

        self.grid_data: dict[tuple[int, int], str] = {}
        self.creator_visible: bool = False

        self._build_layout()
        self._populate_blueprint_tree()
        self._draw_grid()

    def _build_layout(self) -> None:
        # Menüleiste
        top_bar = ttk.Frame(self.root, padding=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tool_frame = ttk.LabelFrame(top_bar, text=" Modus ", padding=4)
        tool_frame.pack(side=tk.LEFT, padx=5)

        self.tool_var = tk.StringVar(value=self.active_tool)
        ttk.Radiobutton(tool_frame, text="Pinsel", value="brush", variable=self.tool_var, command=self._on_tool_changed).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(tool_frame, text="Radierer", value="eraser", variable=self.tool_var, command=self._on_tool_changed).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(tool_frame, text="Pipette", value="picker", variable=self.tool_var, command=self._on_tool_changed).pack(side=tk.LEFT, padx=3)

        action_frame = ttk.Frame(top_bar, padding=4)
        action_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Button(action_frame, text="Neu leeren", command=self._clear_world).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="Labyrinth-Vorlage", command=self._generate_default_labyrinth).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="Welt laden", command=self._load_world_dialog).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="Welt speichern", command=self._save_world_dialog).pack(side=tk.LEFT, padx=3)

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Linke Seitenleiste
        self.sidebar = ttk.Frame(main_paned, width=360, padding=6)
        main_paned.add(self.sidebar, weight=0)

        # Filter
        search_frame = ttk.Frame(self.sidebar)
        search_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._populate_blueprint_tree())
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Baumansicht
        tree_frame = ttk.Frame(self.sidebar)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)

        self.tree = ttk.Treeview(tree_frame, columns=("char", "type"), selectmode="browse", show="tree headings")
        self.tree.heading("#0", text="Entität / Objekt")
        self.tree.heading("char", text="Icon")
        self.tree.heading("type", text="Typ")
        self.tree.column("#0", width=190)
        self.tree.column("char", width=40, anchor=tk.CENTER)
        self.tree.column("type", width=80, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selected)

        # Details zum aktiven Objekt
        self.info_label = ttk.Label(self.sidebar, text="Aktiv: Wand", relief=tk.GROOVE, padding=4, wraplength=320)
        self.info_label.pack(side=tk.TOP, fill=tk.X, pady=4)

        # Ausklappbare Sektion: Objektschöpfer
        self.toggle_creator_btn = ttk.Button(
            self.sidebar,
            text="[+] Neues Objekt definieren  ▼",
            command=self._toggle_creator_panel,
        )
        self.toggle_creator_btn.pack(side=tk.TOP, fill=tk.X, pady=4)

        self.creator_frame = ttk.LabelFrame(self.sidebar, text=" Objekt-Designer ", padding=6)
        self._build_creator_form()

        # Canvas-Bereich
        canvas_container = ttk.Frame(main_paned)
        main_paned.add(canvas_container, weight=1)

        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            canvas_container,
            width=self.width * self.CELL_SIZE,
            height=self.height * self.CELL_SIZE,
            bg="#1A202C",
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
            scrollregion=(0, 0, self.width * self.CELL_SIZE, self.height * self.CELL_SIZE),
        )
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<B3-Motion>", self._on_canvas_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        self.status_var = tk.StringVar(value="Bereit.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_creator_form(self) -> None:
        """Baut die Formularfelder für die Definition neuer Blueprints auf."""
        grid_opts = {"sticky": tk.W, "pady": 2}

        ttk.Label(self.creator_frame, text="Name:").grid(row=0, column=0, **grid_opts)
        self.ent_name_var = tk.StringVar(value="Giftiger Busch")
        ttk.Entry(self.creator_frame, textvariable=self.ent_name_var, width=22).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.creator_frame, text="ID (Eindeutig):").grid(row=1, column=0, **grid_opts)
        self.ent_id_var = tk.StringVar(value="poison_bush")
        ttk.Entry(self.creator_frame, textvariable=self.ent_id_var, width=22).grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.creator_frame, text="Typ-Schlüssel:").grid(row=2, column=0, **grid_opts)
        self.ent_type_var = tk.StringVar(value="bush")
        ttk.Entry(self.creator_frame, textvariable=self.ent_type_var, width=22).grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.creator_frame, text="Kategorie:").grid(row=3, column=0, **grid_opts)
        self.ent_cat_var = tk.StringVar(value="obstacle")
        cat_combo = ttk.Combobox(
            self.creator_frame,
            textvariable=self.ent_cat_var,
            values=["obstacle", "entity", "agent"],
            state="readonly",
            width=20,
        )
        cat_combo.grid(row=3, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.creator_frame, text="Symbol (1 Zeichen):").grid(row=4, column=0, **grid_opts)
        self.ent_char_var = tk.StringVar(value="*")
        ttk.Entry(self.creator_frame, textvariable=self.ent_char_var, width=5).grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.creator_frame, text="Farbe:").grid(row=5, column=0, **grid_opts)
        color_row = ttk.Frame(self.creator_frame)
        color_row.grid(row=5, column=1, sticky=tk.W, pady=2)
        self.ent_color_var = tk.StringVar(value="#805AD5")
        self.color_preview = tk.Label(color_row, bg=self.ent_color_var.get(), width=3, relief=tk.RIDGE)
        self.color_preview.pack(side=tk.LEFT, padx=2)
        ttk.Button(color_row, text="Wählen...", command=self._pick_color).pack(side=tk.LEFT, padx=2)

        self.ent_obstacle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.creator_frame, text="Blockiert Weg (Obstacle)", variable=self.ent_obstacle_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.ent_conversational_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.creator_frame, text="Ist dialogfähig", variable=self.ent_conversational_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        btn_row = ttk.Frame(self.creator_frame)
        btn_row.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=6)
        ttk.Button(btn_row, text="Erstellen & Speichern", command=self._save_new_blueprint).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_row, text="Abbrechen", command=self._toggle_creator_panel).pack(side=tk.LEFT, padx=2)

    def _toggle_creator_panel(self) -> None:
        """Blendet das Creator-Panel ein oder aus."""
        if self.creator_visible:
            self.creator_frame.pack_forget()
            self.toggle_creator_btn.config(text="[+] Neues Objekt definieren  ▼")
            self.creator_visible = False
        else:
            self.creator_frame.pack(side=tk.TOP, fill=tk.X, pady=4, before=self.toggle_creator_btn)
            self.toggle_creator_btn.config(text="[-] Objekt-Designer schließen  ▲")
            self.creator_visible = True

    def _pick_color(self) -> None:
        selected = colorchooser.askcolor(color=self.ent_color_var.get(), title="Objektfarbe wählen")
        if selected and selected[1]:
            hex_color = selected[1]
            self.ent_color_var.set(hex_color)
            self.color_preview.config(bg=hex_color)

    def _save_new_blueprint(self) -> None:
        bp_id = self.ent_id_var.get().strip()
        name = self.ent_name_var.get().strip()
        entity_type = self.ent_type_var.get().strip()
        category = self.ent_cat_var.get()
        raw_char = self.ent_char_var.get().strip()
        color = self.ent_color_var.get().strip()
        is_obstacle = self.ent_obstacle_var.get()
        is_conversational = self.ent_conversational_var.get()

        if not bp_id or not name or not entity_type:
            messagebox.showerror("Fehler", "ID, Name und Typ-Schlüssel dürfen nicht leer sein.")
            return

        if not raw_char:
            messagebox.showerror("Fehler", "Es muss ein Zeichen angegeben werden.")
            return

        char = raw_char[0]

        # Icon-Unikatprüfung: Prüfen, ob das Zeichen bereits vergeben ist
        for existing in self.blueprints:
            if existing.char == char and existing.id != bp_id:
                messagebox.showerror(
                    "Icon bereits vergeben",
                    f"Das Zeichen '{char}' wird bereits von '{existing.name}' ({existing.id}) verwendet.\n"
                    "Für die spätere Sprite-Zuordnung muss jedes Objekt ein eindeutiges Zeichen besitzen.",
                )
                return

        new_bp = EntityBlueprint(
            id=bp_id,
            name=name,
            category=category,
            entity_type=entity_type,
            is_conversational=is_conversational,
            is_obstacle=is_obstacle,
            color=color,
            char=char,
            default_properties={"agent_id": bp_id, "name": name} if category == "agent" else {},
        )

        self.blueprint_repo.add_or_update(new_bp)
        self.blueprints = self.blueprint_repo.get_all()

        self._populate_blueprint_tree()
        if self.tree.exists(bp_id):
            self.tree.selection_set(bp_id)
            self.tree.see(bp_id)
            self._on_tree_selected(None)

        self._toggle_creator_panel()
        self.status_var.set(f"Blueprint '{name}' mit Icon '{char}' erfolgreich gespeichert.")

    def _on_tool_changed(self) -> None:
        self.active_tool = self.tool_var.get()
        self.status_var.set(f"Aktives Werkzeug: {self.active_tool}")

    def _populate_blueprint_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        query = self.search_var.get().strip().lower()

        categories: dict[str, list] = {}
        for bp in self.blueprints:
            if query and query not in bp.name.lower() and query not in bp.entity_type.lower():
                continue
            cat = self.CATEGORY_NAMES.get(bp.category, "4. Sonstiges")
            categories.setdefault(cat, []).append(bp)

        for cat_name in sorted(categories.keys()):
            cat_node = self.tree.insert("", tk.END, text=cat_name, open=True)
            for bp in categories[cat_name]:
                self.tree.insert(
                    cat_node,
                    tk.END,
                    iid=bp.id,
                    text=bp.name,
                    values=(bp.char, bp.entity_type),
                )

        if self.active_blueprint_id and self.tree.exists(self.active_blueprint_id):
            self.tree.selection_set(self.active_blueprint_id)

    def _on_tree_selected(self, event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        bp = self.blueprint_repo.get_by_id(item_id)
        if bp:
            self.active_blueprint_id = bp.id
            self.active_tool = "brush"
            self.tool_var.set("brush")
            conv_text = "Ja" if bp.is_conversational else "Nein"
            obs_text = "Ja" if bp.is_obstacle else "Nein"
            self.info_label.config(
                text=f"Aktiv: {bp.name}\nTyp: {bp.entity_type} | Icon: {bp.char}\nBlockiert: {obs_text} | Dialogfähig: {conv_text}"
            )

    def _draw_grid(self) -> None:
        self.canvas.delete("all")
        for x in range(self.width):
            for y in range(self.height):
                self._draw_cell(x, y)

    def _draw_cell(self, x: int, y: int) -> None:
        x1 = x * self.CELL_SIZE
        y1 = y * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE

        bp_id = self.grid_data.get((x, y))
        color = "#1A202C"
        char = ""
        if bp_id:
            bp = self.blueprint_repo.get_by_id(bp_id)
            if bp:
                color = bp.color
                char = bp.char

        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color,
            outline="#2D3748",
            tags=f"cell_{x}_{y}",
        )
        if char:
            self.canvas.create_text(
                x1 + self.CELL_SIZE // 2,
                y1 + self.CELL_SIZE // 2,
                text=char,
                fill="#FFFFFF",
                font=("Monospace", 9, "bold"),
                tags=f"text_{x}_{y}",
            )

    def _update_cell(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        self.canvas.delete(f"cell_{x}_{y}")
        self.canvas.delete(f"text_{x}_{y}")
        self._draw_cell(x, y)

    def _apply_interaction(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        if self.active_tool == "eraser":
            self.grid_data.pop((x, y), None)
            self._update_cell(x, y)

        elif self.active_tool == "picker":
            bp_id = self.grid_data.get((x, y))
            if bp_id and self.tree.exists(bp_id):
                self.tree.selection_set(bp_id)
                self.tree.see(bp_id)
                self._on_tree_selected(None)

        elif self.active_tool == "brush":
            bp = self.blueprint_repo.get_by_id(self.active_blueprint_id)
            if not bp:
                return

            if bp.category == "agent":
                for pos, existing_id in list(self.grid_data.items()):
                    if existing_id == bp.id:
                        del self.grid_data[pos]
                        self._update_cell(pos[0], pos[1])

            self.grid_data[(x, y)] = bp.id
            self._update_cell(x, y)

    def _on_canvas_click(self, event: tk.Event) -> None:
        x = int(self.canvas.canvasx(event.x) // self.CELL_SIZE)
        y = int(self.canvas.canvasy(event.y) // self.CELL_SIZE)
        self._apply_interaction(x, y)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        x = int(self.canvas.canvasx(event.x) // self.CELL_SIZE)
        y = int(self.canvas.canvasy(event.y) // self.CELL_SIZE)
        self._apply_interaction(x, y)

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        x = int(self.canvas.canvasx(event.x) // self.CELL_SIZE)
        y = int(self.canvas.canvasy(event.y) // self.CELL_SIZE)
        if (x, y) in self.grid_data:
            del self.grid_data[(x, y)]
            self._update_cell(x, y)

    def _on_mouse_move(self, event: tk.Event) -> None:
        x = int(self.canvas.canvasx(event.x) // self.CELL_SIZE)
        y = int(self.canvas.canvasy(event.y) // self.CELL_SIZE)
        item = self.grid_data.get((x, y), "Frei")
        self.status_var.set(f"Position: ({x}, {y}) | Inhalt: {item} | Modus: {self.active_tool}")

    def _clear_world(self) -> None:
        if messagebox.askyesno("Grid leeren", "Soll die gesamte Welt zurückgesetzt werden?"):
            self.grid_data.clear()
            self._draw_grid()

    def _generate_default_labyrinth(self) -> None:
        self.grid_data.clear()

        # Äußere Grenzwände
        for x in range(self.width):
            self.grid_data[(x, 0)] = "wall"
            self.grid_data[(x, self.height - 1)] = "wall"
        for y in range(self.height):
            self.grid_data[(0, y)] = "wall"
            self.grid_data[(self.width - 1, y)] = "wall"

        # Barriere 1
        for y in range(1, 36):
            self.grid_data[(15, y)] = "wall"
        for x in range(1, 11):
            self.grid_data[(x, 28)] = "wall"

        # Barriere 2
        for y in range(9, 44):
            self.grid_data[(30, y)] = "wall"
        for x in range(16, 26):
            self.grid_data[(x, 20)] = "wall"

        # Barriere 3
        for y in range(1, 21):
            self.grid_data[(45, y)] = "wall"
        self.grid_data[(45, 21)] = "wall"
        self.grid_data[(45, 23)] = "wall"
        for y in range(24, 36):
            self.grid_data[(45, y)] = "wall"

        for x in range(38, 45):
            if x != 44:
                self.grid_data[(x, 20)] = "wall"
            self.grid_data[(x, 24)] = "wall"

        # Barriere 4
        for y in range(9, 44):
            self.grid_data[(60, y)] = "wall"
        for x in range(48, 57):
            self.grid_data[(x, 22)] = "wall"

        # Barriere 5
        for y in range(1, 36):
            self.grid_data[(75, y)] = "wall"
        for x in range(76, 86):
            self.grid_data[(x, 15)] = "wall"

        for pos in [(7, 10), (22, 35), (36, 12), (52, 30), (68, 18), (82, 32)]:
            self.grid_data[pos] = "boulder"

        self.grid_data[(45, 22)] = "rock"
        self.grid_data[(2, 22)] = "agent_alice"
        self.grid_data[(87, 22)] = "target_east"

        self._draw_grid()

    def _to_world_definition(self, name: str) -> WorldDefinition:
        obstacles: list[Position] = []
        entities: list[PlacedEntityData] = []
        agents: list[PlacedAgentData] = []

        target_pos = Position(87, 22)
        target_name = "Ost-Tor"

        for (x, y), bp_id in self.grid_data.items():
            bp = self.blueprint_repo.get_by_id(bp_id)
            if bp and bp.entity_type == "target":
                target_pos = Position(x, y)
                target_name = bp.name
                break

        ent_counter = 1
        for (x, y), bp_id in self.grid_data.items():
            bp = self.blueprint_repo.get_by_id(bp_id)
            if not bp:
                continue

            pos = Position(x, y)
            if bp.category == "obstacle" and bp.entity_type != "target":
                obstacles.append(pos)
            elif bp.category == "entity":
                entities.append(
                    PlacedEntityData(
                        id=f"{bp.entity_type}_{ent_counter}",
                        name=bp.name,
                        blueprint_id=bp.id,
                        position=pos,
                        entity_type=bp.entity_type,
                        is_conversational=bp.is_conversational,
                    )
                )
                ent_counter += 1
            elif bp.category == "agent":
                props = bp.default_properties
                agents.append(
                    PlacedAgentData(
                        id=props.get("agent_id", "1"),
                        name=props.get("name", bp.name),
                        position=pos,
                        target_position=target_pos,
                        destination_name=target_name,
                    )
                )

        return WorldDefinition(
            name=name,
            width=self.width,
            height=self.height,
            obstacles=obstacles,
            entities=entities,
            agents=agents,
        )

    def _save_world_dialog(self) -> None:
        name = simpledialog.askstring("Speichern", "Name der Karte:", initialvalue="labyrinth")
        if not name:
            return
        world = self._to_world_definition(name)
        saved_path = self.world_repo.save(world, name)
        messagebox.showinfo("Gespeichert", f"Welt gespeichert unter:\n{saved_path}")

    def _load_world_dialog(self) -> None:
        worlds = self.world_repo.list_worlds()
        if not worlds:
            messagebox.showwarning("Hinweis", "Keine gespeicherten Welten vorhanden.")
            return

        name = simpledialog.askstring("Laden", f"Vorhandene Welten:\n{', '.join(worlds)}\n\nKartenname:")
        if not name:
            return
        try:
            world = self.world_repo.load(name)
            self.width = world.width
            self.height = world.height
            self.grid_data.clear()

            for obs in world.obstacles:
                self.grid_data[(obs.x, obs.y)] = "wall"
            for ent in world.entities:
                self.grid_data[(ent.position.x, ent.position.y)] = ent.blueprint_id
            for ag in world.agents:
                bp_id = "agent_bob" if ag.id == "2" else "agent_alice"
                self.grid_data[(ag.position.x, ag.position.y)] = bp_id
                if ag.target_position:
                    self.grid_data[(ag.target_position.x, ag.target_position.y)] = "target_east"

            self._draw_grid()
            messagebox.showinfo("Geladen", f"Welt '{world.name}' wurde geladen.")
        except Exception as err:
            messagebox.showerror("Fehler", f"Fehler beim Laden: {err}")

    def _open_char_picker(self) -> None:
        """Öffnet eine programmatisch generierte ASCII- und CP437-Zeichentabelle."""
        dialog = tk.Toplevel(self.root)
        dialog.title("ASCII- & CP437-Zeichentabelle")
        dialog.geometry("620x520")
        dialog.transient(self.root)
        dialog.grab_set()

        current_id = self.ent_id_var.get().strip()
        used_chars = {bp.char: bp.name for bp in self.blueprints if bp.id != current_id}

        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        info_var = tk.StringVar(value="Wähle ein Zeichen aus der Tabelle.")
        ttk.Label(dialog, textvariable=info_var, relief=tk.SUNKEN, anchor=tk.W, padding=4).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        def make_select_cmd(char: str):
            def select():
                self.ent_char_var.set(char)
                dialog.destroy()

            return select

        # 1. Standard-ASCII (Dec 32 bis 126)
        ascii_chars = [(i, chr(i)) for i in range(32, 127)]

        # 2. CP437 Extended ASCII (Symbole & Rahmen, Dec 1 bis 31 und 128 bis 255)
        cp437_chars = []
        for code in list(range(1, 32)) + list(range(128, 256)):
            try:
                char_str = bytes([code]).decode("cp437")
                cp437_chars.append((code, char_str))
            except UnicodeDecodeError:
                continue

        tabs = [
            ("Standard-ASCII (32-126)", ascii_chars),
            ("Erweitert / CP437 (Symbole & Rahmen)", cp437_chars),
        ]

        for tab_name, char_set in tabs:
            tab_frame = ttk.Frame(notebook, padding=6)
            notebook.add(tab_frame, text=tab_name)

            canvas = tk.Canvas(tab_frame, bg="#FFFFFF")
            v_scroll = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=canvas.yview)
            scroll_content = ttk.Frame(canvas)

            scroll_content.bind(
                "<Configure>",
                lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")),
            )
            canvas.create_window((0, 0), window=scroll_content, anchor="nw")
            canvas.configure(yscrollcommand=v_scroll.set)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            cols_per_row = 16
            for idx, (dec_val, ch) in enumerate(char_set):
                row = idx // cols_per_row
                col = idx % cols_per_row

                is_used = ch in used_chars
                owner = used_chars.get(ch, "")

                btn = tk.Button(
                    scroll_content,
                    text=ch if ch != " " else "␣",
                    width=2,
                    height=1,
                    font=("Monospace", 10, "bold"),
                    relief=tk.RIDGE,
                )

                hex_val = hex(dec_val).upper()

                if is_used:
                    btn.config(state=tk.DISABLED, bg="#CBD5E0", disabledforeground="#E53E3E")
                    btn.bind("<Enter>", lambda _, o=owner, c=ch, h=hex_val, d=dec_val: info_var.set(
                        f"'{c}' (Dec: {d}, Hex: {h}) ist belegt durch: {o}"
                    ))
                    btn.bind("<Leave>", lambda _: info_var.set("Wähle ein Zeichen aus der Tabelle."))
                else:
                    btn.config(command=make_select_cmd(ch), bg="#F7FAFC", fg="#2D3748")
                    btn.bind("<Enter>", lambda _, c=ch, h=hex_val, d=dec_val: info_var.set(
                        f"'{c}' (Dec: {d}, Hex: {h}) - Verfügbar"
                    ))
                    btn.bind("<Leave>", lambda _: info_var.set("Wähle ein Zeichen aus der Tabelle."))

                btn.grid(row=row, column=col, padx=1, pady=1)


def main() -> None:
    root = tk.Tk()
    app = WorldEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()