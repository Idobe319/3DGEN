import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import threading
import os
import sys
import time
import trimesh
import numpy as np
import random
import re

# sanitize log text (remove ANSI escape sequences and non-printable chars)
def _sanitize_log(s: str) -> str:
    if not s:
        return s
    # Remove ANSI escape sequences (CSI and similar)
    s = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', s)
    # Remove other control characters except common whitespace
    s = ''.join(ch for ch in s if ch.isprintable() or ch in '\t\n\r')
    return s

# --- SAFETY CHECK FOR MATPLOTLIB ---
try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    matplotlib = None

# --- ULTIMATE THEME CONFIG ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

C_BG = "#050505"  # Void Black
C_PANEL = "#0a0b10"  # Dark Metal
C_ACCENT = "#00f2ff"  # Cyber Cyan
C_ACCENT_DIM = "#005f66"
C_TEXT = "#e0e0e0"
C_WARN = "#ff3333"
FONT_MAIN = ("Roboto Medium", 11)
FONT_MONO = ("Consolas", 10)
FONT_HEAD = ("Impact", 24)
POINT_SIZE = 8


class TypewriterText(ctk.CTkTextbox):
    """תיבת טקסט עם אפקט כתיבה בזמן אמת"""

    def write_effect(self, text):
        self.configure(state="normal")
        self.insert("end", "> ")
        self.see("end")

        # כתיבה אות אחרי אות (מהיר)
        for char in text:
            self.insert("end", char)
            self.see("end")
            self.update_idletasks()
            time.sleep(0.005)  # מהירות ההקלדה

        self.insert("end", "\n")
        self.configure(state="disabled")


class LiveGraph(ctk.CTkFrame):
    """גרף טלמטריה חי"""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#000000", **kwargs)
        self.canvas = tk.Canvas(self, bg="black", height=60, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.data = [30] * 50
        self.running = True
        self.animate()

    def animate(self):
        if not self.running:
            return

        # יצירת דאטה רנדומלי שנראה כמו עומס GPU
        new_val = self.data[-1] + random.randint(-10, 10)
        new_val = max(10, min(50, new_val))
        self.data.append(new_val)
        self.data.pop(0)

        self.canvas.delete("all")

        # ציור הגרף
        w = self.winfo_width()
        h = 60
        step = w / len(self.data)

        points = []
        for i, val in enumerate(self.data):
            x = i * step
            y = h - val
            points.append(x)
            points.append(y)

        if len(points) > 4:
            self.canvas.create_line(points, fill=C_ACCENT, width=1.5, smooth=True)
            # אפקט מילוי מתחת לקו
            points.extend([w, h, 0, h])
            self.canvas.create_polygon(points, fill=C_ACCENT_DIM, outline="")

        self.after(100, self.animate)


class ScanningImage(ctk.CTkFrame):
    """תצוגת תמונה עם אפקט לייזר סורק"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.image_item = None
        self.scan_line = None
        self.tk_img = None
        self.scan_y = 0
        self.scanning = False
        self.img_h = 0
        self.img_w = 0

    def set_image(self, img_path):
        pil_img = Image.open(img_path)

        # התאמה לגודל הקנבס
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10:
            cw = 400
        if ch < 10:
            ch = 400

        ratio = min(cw / pil_img.width, ch / pil_img.height)
        nw = int(pil_img.width * ratio)
        nh = int(pil_img.height * ratio)
        self.img_w, self.img_h = nw, nh

        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        # מרכוז
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.image_item = self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

        # יצירת קו הסריקה
        self.scan_line = self.canvas.create_line(x, y, x + nw, y, fill=C_ACCENT, width=2)
        self.scan_y = y
        self.start_y = y
        self.end_y = y + nh
        self.start_x = x
        self.end_x = x + nw

        self.scanning = True
        self.animate_scan()

    def animate_scan(self):
        if not self.scanning:
            return

        self.scan_y += 2
        if self.scan_y > self.end_y:
            self.scan_y = self.start_y

        # Guard against scan_line being None (type checkers warn)
        scan_line = getattr(self, "scan_line", None)
        if scan_line is not None:
            self.canvas.coords(scan_line, self.start_x, self.scan_y, self.end_x, self.scan_y)

        # אפקט זוהר מתחת לקו
        # (אופציונלי למתקדמים, כרגע נשמור פשוט לביצועים)

        self.after(20, self.animate_scan)

    def stop_scan(self):
        self.scanning = False
        self.canvas.delete("all")


class HologramView(ctk.CTkFrame):
    """תצוגת תלת ממד אולטימטיבית"""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="black", **kwargs)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        # Use the public API to set facecolor (avoids attribute access warnings from Pylance)
        self.figure.set_facecolor("black")
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Import typing.Any and use it for the axes object to avoid fragile static type checks
        from typing import Any
        # Keep the runtime behavior: add a 3D subplot and treat its type as Any for the typechecker
        self.ax: Any = self.figure.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.ax.grid(False)

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.points = None
        self.angle = 0
        self.mode = "IDLE"

        # טקסט הולוגרפי (use standard .text which is well-typed)
        # 3D axes text requires a z coordinate; provide z=0 to avoid missing-argument errors
        # Use explicit keyword 's' to match the matplotlib.text signature
        self.txt = self.ax.text(x=0.5, y=0.5, z=0, s="NO SIGNAL", transform=self.ax.transAxes, color="#333333", ha="center", family="monospace", fontsize=14)

        self.animate()

    def load_mesh(self, path):
        try:
            mesh = trimesh.load(path, process=False)
            verts = None
            # If we got a Trimesh directly, use it. If a Scene, pick the first geometry.
            if isinstance(mesh, trimesh.Trimesh):
                verts = mesh.vertices
            elif isinstance(mesh, trimesh.Scene):
                if mesh.geometry:
                    first = next(iter(mesh.geometry.values()))
                    if isinstance(first, trimesh.Trimesh):
                        verts = first.vertices

            if verts is not None:
                # דגימה חכמה
                if len(verts) > 3000:
                    idx = np.random.choice(len(verts), 3000, replace=False)
                    verts = verts[idx]

                # נרמול
                centroid = np.mean(verts, axis=0)
                verts = verts - centroid
                scale = np.max(np.abs(verts))
                if scale != 0:
                    verts = verts / scale

                self.points = verts
                self.mode = "MESH"
                try:
                    self.txt.set_text("")
                except Exception:
                    pass
        except Exception:
            pass

    def animate(self):
        self.angle = (self.angle + 0.5) % 360
        self.ax.view_init(elev=10, azim=self.angle)

        self.ax.clear()
        self.ax.axis("off")

        if self.mode == "COMPUTING":
            # אפקט חלקיקים מתפוצצים
            n = 200
            theta = np.random.uniform(0, 2 * np.pi, n)
            phi = np.random.uniform(0, np.pi, n)
            r = 0.5 + 0.1 * np.sin(time.time() * 5)  # פעימה

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Convert to native lists & use integer size for scatter
            # Use keyword 'zs' so static analyzers recognize this as a 3D scatter
            self.ax.scatter(x, y, zs=z, c=C_ACCENT, s=POINT_SIZE, alpha=0.5)
            # Use explicit keyword 's' for the text string argument
            # Provide explicit z coordinate for 3D text to satisfy Axes3D API
            self.ax.text(x=0.5, y=0.1, z=0, s="BUILDING NEURAL VOXELS...", transform=self.ax.transAxes, color=C_ACCENT, ha="center", family="monospace")

        elif self.mode == "MESH" and self.points is not None:
            # מודל מסתובב - החלפת צירים ליישור
            # Use 'zs=' keyword for the third coordinate to satisfy type checkers
            self.ax.scatter(self.points[:, 0], self.points[:, 2], zs=self.points[:, 1], c=C_ACCENT, s=POINT_SIZE, alpha=0.8, linewidths=0)

        else:
            # Use .text which is supported on Axes and typed for Pylance
            # Use explicit z and keyword 's' for clarity with 3D axes
            self.ax.text(x=0.5, y=0.5, z=0, s="SYSTEM IDLE", transform=self.ax.transAxes, color="#333")

        self.canvas.draw()
        self.after(30, self.animate)


class NeoForgeUltimate(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeoForge 3D // ULTIMATE EDITION")
        self.geometry("1600x900")
        self.configure(fg_color=C_BG)

        # משתנים
        self.selected_file = ""
        self.start_time = 0
        self.is_running = False
        self.last_update = 0

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(1, weight=1)  # Center expands
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Left)
        self.sidebar = ctk.CTkFrame(self, width=300, fg_color=C_PANEL, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self._build_sidebar()

        # 2. MAIN VIEW (Center)
        self.center = ctk.CTkFrame(self, fg_color="transparent")
        self.center.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        # מסגרת לתצוגה
        self.viewport_frame = ctk.CTkFrame(self.center, fg_color="black", border_width=2, border_color="#222")
        self.viewport_frame.grid(row=0, column=0, sticky="nsew")

        # שכבות תצוגה
        self.scanner = ScanningImage(self.viewport_frame)
        self.scanner.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.hologram = HologramView(self.viewport_frame)
        # ההולוגרמה תעלה מעל הסורק כשצריך

        # 3. TERMINAL (Bottom)
        self.terminal_frame = ctk.CTkFrame(self.center, height=200, fg_color=C_PANEL, corner_radius=0)
        self.terminal_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.terminal_frame.pack_propagate(False)

        self.log_box = TypewriterText(self.terminal_frame, fg_color=C_BG, text_color=C_ACCENT, font=FONT_MONO)
        self.log_box.pack(fill="both", expand=True, padx=1, pady=1)

        # 4. TELEMETRY (Right)
        self.telemetry = ctk.CTkFrame(self, width=250, fg_color=C_PANEL, corner_radius=0)
        self.telemetry.grid(row=0, column=2, sticky="nsew", padx=(2, 0))
        self._build_telemetry()

    def _build_sidebar(self):
        # Header
        f = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        f.pack(pady=40)
        ctk.CTkLabel(f, text="NEOFORGE", font=FONT_HEAD, text_color="white").pack()
        ctk.CTkLabel(f, text="GENERATION ENGINE V3", font=FONT_MONO, text_color=C_ACCENT).pack()

        # Controls
        self.btn_load = ctk.CTkButton(
            self.sidebar,
            text=" // LOAD SOURCE DATA",
            command=self.load_file,
            fg_color="transparent",
            border_width=1,
            border_color=C_ACCENT,
            text_color=C_ACCENT,
            font=FONT_MONO,
            hover_color="#002222",
            height=40,
        )
        self.btn_load.pack(fill="x", padx=20, pady=20)

        # Settings Group
        lbl = ctk.CTkLabel(self.sidebar, text="CONFIGURATION", font=("Impact", 16), text_color="#555", anchor="w")
        lbl.pack(fill="x", padx=20, pady=(20, 5))

        self.slider = ctk.CTkSlider(
            self.sidebar, from_=1000, to=20000, progress_color=C_ACCENT, button_color="white", command=self.update_poly
        )
        self.slider.set(5000)
        self.slider.pack(fill="x", padx=20, pady=5)

        self.lbl_poly = ctk.CTkLabel(self.sidebar, text="5000 POLYGONS", font=FONT_MONO, text_color="white")
        self.lbl_poly.pack()

        self.sw_bake = ctk.CTkSwitch(self.sidebar, text="TEXTURE BAKING", progress_color=C_ACCENT, font=FONT_MONO)
        self.sw_bake.select()
        self.sw_bake.pack(anchor="w", padx=20, pady=10)

        self.sw_hq = ctk.CTkSwitch(self.sidebar, text="FORCE HIGH-RES", progress_color=C_ACCENT, font=FONT_MONO)
        self.sw_hq.select()
        self.sw_hq.pack(anchor="w", padx=20, pady=10)

        # Fast preview (low-res, faster generation)
        self.sw_preview = ctk.CTkSwitch(self.sidebar, text="FAST PREVIEW (LOW-RES)", progress_color=C_ACCENT, font=FONT_MONO)
        self.sw_preview.pack(anchor="w", padx=20, pady=6)

        # Action
        self.btn_run = ctk.CTkButton(
            self.sidebar,
            text="INITIATE SEQUENCE",
            command=self.start,
            fg_color=C_ACCENT,
            text_color="black",
            font=("Impact", 18),
            height=60,
            hover_color="white",
        )
        self.btn_run.pack(side="bottom", fill="x", padx=20, pady=40)

    def _build_telemetry(self):
        ctk.CTkLabel(self.telemetry, text="LIVE TELEMETRY", font=("Impact", 16), text_color="#555").pack(pady=20)

        # GPU Graph
        ctk.CTkLabel(self.telemetry, text="GPU LOAD", font=FONT_MONO, anchor="w").pack(fill="x", padx=10)
        # Make the graph frame a fixed height; don't pass `height` to `pack` (pack doesn't accept it)
        self.graph = LiveGraph(self.telemetry, height=60)
        self.graph.pack(fill="x", padx=10, pady=5)

        # Stats
        self.lbl_time = self._add_stat("ELAPSED TIME", "00:00:00")
        self.lbl_stage = self._add_stat("OPERATION", "STANDBY")
        self.lbl_verts = self._add_stat("VERTICES", "---")

        # Stage progress bar
        self.stage_progress = ctk.CTkProgressBar(self.telemetry, orientation="horizontal", width=200)
        self.stage_progress.set(0.0)
        self.stage_progress.pack(fill="x", padx=10, pady=(5, 2))
        self.lbl_percent = ctk.CTkLabel(self.telemetry, text="0%", font=FONT_MONO, text_color=C_ACCENT)
        self.lbl_percent.pack(anchor="e", padx=10)

        # ETA and GPU
        self.lbl_eta = self._add_stat("ETA", "--:--")
        self.lbl_gpu = self._add_stat("GPU", "N/A")

        # Result Button (Hidden)
        self.btn_open = ctk.CTkButton(
            self.telemetry, text="OPEN VIEWER", command=self.open_file, fg_color="#333", state="disabled"
        )
        self.btn_open.pack(side="bottom", fill="x", padx=10, pady=20)

        # Internal state for progress timing
        self._stage_start_times = {}
        self._last_stage_payload = {}

    def _add_stat(self, title, val):
        f = ctk.CTkFrame(self.telemetry, fg_color="#0f0f0f")
        f.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(f, text=title, font=("Arial", 8, "bold"), text_color="#666").pack(anchor="w", padx=5)
        lbl_val = ctk.CTkLabel(f, text=val, font=("Consolas", 14), text_color=C_ACCENT)
        lbl_val.pack(anchor="e", padx=5)
        return lbl_val

    def update_poly(self, val):
        self.lbl_poly.configure(text=f"{int(val)} POLYGONS")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path:
            self.selected_file = path
            self.hologram.place_forget()  # הסתרת הולוגרמה
            self.scanner.set_image(path)  # הצגת תמונה סורקת
            self.log_box.write_effect(f"SOURCE LOADED: {os.path.basename(path)}")

    def start(self):
        if not self.selected_file:
            self.log_box.write_effect("ERROR: NO SOURCE DATA DETECTED.")
            return

        self.is_running = True
        self.start_time = time.time()
        self.btn_run.configure(state="disabled", text="PROCESSING...")
        self.lbl_stage.configure(text="INITIALIZING")

        # מעבר להולוגרמה של חלקיקים
        self.scanner.stop_scan()
        self.hologram.mode = "COMPUTING"
        self.hologram.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.update_clock()
        self.monitor_thread = threading.Thread(target=self.monitor_backend, daemon=True)
        self.monitor_thread.start()

        threading.Thread(target=self.run_process, daemon=True).start()

    def update_clock(self):
        if self.is_running:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.lbl_time.configure(text=f"00:{mins:02}:{secs:02}")
            self.after(1000, self.update_clock)

    def monitor_backend(self):
        # בודק קבצים ברקע כדי לעדכן את ההולוגרמה
        while self.is_running:
            raw = os.path.join(os.getcwd(), "temp_raw.obj")
            clean = os.path.join(os.getcwd(), "workspace", "clean_quads.obj")

            target = None
            if os.path.exists(clean) and os.path.getmtime(clean) > self.last_update:
                target = clean
            elif os.path.exists(raw) and os.path.getmtime(raw) > self.last_update:
                target = raw

            if target:
                self.last_update = time.time()
                # עדכון ההולוגרמה מה-Main Thread בזהירות
                self.after(0, lambda p=target: self.hologram.load_mesh(p))

            time.sleep(1.5)

    def run_process(self):
        script = os.path.join(os.getcwd(), "run_local.py")
        cmd = [
            sys.executable,
            script,
            "--input",
            self.selected_file,
            "--poly",
            str(int(self.slider.get())),
            "--quality",
            "high",
        ]

        if self.sw_bake.get():
            cmd.append("--bake")
        if self.sw_hq.get():
            cmd.append("--force-high")
        if getattr(self, 'sw_preview', None) and self.sw_preview.get():
            cmd.append('--fast-preview')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        out = process.stdout
        engine_name = None
        if out is not None:
            for line in iter(out.readline, ""):
                if line:
                    clean_line = _sanitize_log(line).strip()
                    self.after(0, lambda t=clean_line: self.log_box.write_effect(t))

                    # זיהוי מנוע (TripoSR/TRELLIS) מתוך הלוג
                    lower = clean_line.lower()
                    if "attempting to use external triposr" in lower or "triposr: running command" in lower:
                        engine_name = "TripoSR"
                    elif "loading ai geometry engine (trellis" in lower or "trellis inference" in lower:
                        engine_name = "TRELLIS"

                    # Parse structured progress JSON if present
                    if clean_line.startswith("PROGRESS_JSON:"):
                        try:
                            import json
                            payload = json.loads(clean_line[len("PROGRESS_JSON:"):].strip())
                            self.after(0, lambda p=payload: self._handle_progress(p))
                        except Exception:
                            pass
                        continue

                    # עדכון סטטוס
                    status = None
                    if "stage 1" in lower:
                        status = "PRE-PROCESS"
                    elif "stage 2" in lower:
                        status = f"AI INFERENCE ({engine_name if engine_name else 'ENGINE'})"
                    elif "stage 3" in lower:
                        status = f"GEOMETRY ({engine_name if engine_name else 'ENGINE'})"
                    elif "stage 4" in lower:
                        status = "UV MAPPING"
                    elif "baking" in lower:
                        status = "BAKING TEXTURES"

                    if status:
                        self.after(0, lambda s=status: self.lbl_stage.configure(text=s))
        else:
            stdout_data, _ = process.communicate()
            if stdout_data:
                for line in stdout_data.splitlines():
                    clean_line = _sanitize_log(line).strip()
                    self.after(0, lambda t=clean_line: self.log_box.write_effect(t))
                    # Parse structured progress JSON if present
                    if clean_line.startswith("PROGRESS_JSON:"):
                        try:
                            import json

                            payload = json.loads(clean_line[len("PROGRESS_JSON:"):].strip())
                            self.after(0, lambda p=payload: self._handle_progress(p))
                        except Exception:
                            pass
                        continue
                    lower = clean_line.lower()
                    status = None
                    if "stage 1" in lower:
                        status = "PRE-PROCESS"
                    elif "stage 2" in lower:
                        status = "AI INFERENCE"
                    elif "stage 3" in lower:
                        status = "GEOMETRY"
                    elif "stage 4" in lower:
                        status = "UV MAPPING"
                    elif "baking" in lower:
                        status = "BAKING TEXTURES"
                    if status:
                        self.after(0, lambda s=status: self.lbl_stage.configure(text=s))

        process.wait()

        self.is_running = False
        self.after(0, self.on_finish)

    def _format_bytes(self, b: int | None) -> str:
        if b is None:
            return "N/A"
        try:
            mb = int(b) / (1024 * 1024)
            return f"{mb:.1f} MB"
        except Exception:
            return str(b)

    def _handle_progress(self, payload: dict) -> None:
        # Expected payload: {ts, stage, percent?, details?}
        stage = payload.get('stage') or payload.get('phase') or 'unknown'
        percent = payload.get('percent')
        details = payload.get('details') or {}

        # Update stage name
        self.lbl_stage.configure(text=stage.upper())

        # Track start time for ETA calculation
        now = time.time()
        if stage not in self._stage_start_times:
            self._stage_start_times[stage] = now

        # Update progress bar
        if isinstance(percent, (int, float)):
            p = max(0.0, min(100.0, float(percent)))
            self.stage_progress.set(p / 100.0)
            self.lbl_percent.configure(text=f"{int(p)}%")

            # ETA estimation
            start = self._stage_start_times.get(stage, now)
            elapsed = now - start
            if p > 0.001:
                total_est = elapsed / (p / 100.0)
                eta = max(0.0, total_est - elapsed)
                mins = int(eta // 60)
                secs = int(eta % 60)
                self.lbl_eta.configure(text=f"{mins:02}:{secs:02}")
            else:
                self.lbl_eta.configure(text="--:--")
        else:
            # Indeterminate / unknown progress
            self.stage_progress.set(0.0)
            self.lbl_percent.configure(text="--%")
            self.lbl_eta.configure(text="--:--")

        # Update vertices if provided
        if isinstance(details, dict):
            verts = details.get('target_vertices') or details.get('vertices')
            if verts is not None:
                try:
                    self.lbl_verts.configure(text=str(int(verts)))
                except Exception:
                    self.lbl_verts.configure(text=str(verts))

            # GPU info
            gpu = details.get('gpu')
            if isinstance(gpu, dict):
                mem_alloc = gpu.get('memory_allocated')
                mem_res = gpu.get('memory_reserved')
                name = gpu.get('device') or gpu.get('name')
                self.lbl_gpu.configure(text=f"{name} {self._format_bytes(mem_alloc)} / {self._format_bytes(mem_res)}")

        # Cache last payload
        self._last_stage_payload[stage] = payload

    def on_finish(self):
        self.btn_run.configure(state="normal", text="INITIATE SEQUENCE")
        self.lbl_stage.configure(text="COMPLETE")
        self.btn_open.configure(state="normal", fg_color=C_ACCENT, text_color="black")
        self.log_box.write_effect(">>> PROCESS COMPLETED SUCCESSFULLY.")

        # פתיחה אוטומטית
        self.open_file()

    def open_file(self):
        final = os.path.join(os.getcwd(), "workspace", "clean_quads_uv_textured.glb")
        if os.path.exists(final):
            os.startfile(final)


if __name__ == "__main__":
    app = NeoForgeUltimate()
    app.mainloop()
