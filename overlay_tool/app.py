"""Tkinter app: fixed board outline with auto-registered photo overlays.

Two plots (top / bottom side). The DXF outline never moves; each opened
photo is segmented, aligned (rotation / mirror / scale / perspective) and
warped into the outline's mm frame, then shown under the outline.

The last selected outline / top / bottom files are remembered in a JSON
settings file and restored automatically on the next launch.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .bom import BomItem, load_bom
from .outline import Outline, load_outline
from .pnp import Part, load_pnp
from .presence import PresenceResult, check_presence
from . import distortion, presence_cnn
from .register import load_photo, register_photo

IOU_WARN = 0.90  # below this, flag the fit as questionable

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".pcb_overlay_tool.json")

IMAGE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.dng"),
    ("All files", "*.*"),
]
DXF_TYPES = [("DXF board outline", "*.dxf"), ("All files", "*.*")]
PNP_TYPES = [("Pick & place", "*.txt *.csv"), ("All files", "*.*")]
BOM_TYPES = [("BOM", "*.xlsx *.xls"), ("All files", "*.*")]

SIDE_KEYS = ("top", "bottom")
SIDE_NAMES = ("Top side", "Bottom side (bottom view)")

# The bottom plot is a true bottom view: photo as shot (readable), outline
# and component frames mirrored (x -> -x) from the top-view frame.
SIDE_XSIGN = (1.0, -1.0)


class OverlayToolbar(NavigationToolbar2Tk):
    """Toolbar whose Home button restores the full-board view, regardless of
    how the plots were zoomed/panned (toolbar, wheel or left-drag)."""

    def __init__(self, canvas, window, app: "OverlayApp"):
        self._app = app
        super().__init__(canvas, window)

    def home(self, *args):
        self._app.reset_view()


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass  # remembering files is best-effort


class OverlayApp:
    def __init__(self, root: tk.Tk, dxf_override: str | None = None):
        self.root = root
        self.settings = load_settings()
        self.outline: Outline | None = None
        root.title("PCB Photo Overlay")
        root.geometry("1000x900")

        self.fig = Figure(figsize=(9, 8), constrained_layout=True)
        self.axes = self.fig.subplots(1, 2)
        self.images = [None, None]  # AxesImage per side
        self.outline_artists = [None, None]
        self.parts: list[Part] = []
        self.bom: dict[str, BomItem] | None = None
        self.frame_artists: list[list] = [[], []]  # LineCollections per side
        self.warped: list = [None, None]           # registered photo per side
        self.reg_info: list = [None, None]         # (path, homography, width)
        self.presence: list[dict[str, PresenceResult]] = [{}, {}]
        self.absent_artists: list[list] = [[], []]
        for ax, name in zip(self.axes, SIDE_NAMES):
            ax.set_title(name)
            ax.set_xlabel("mm")
            # datalim: the axes box always fills its half of the window and
            # zooming pads the data range instead of shrinking the box
            ax.set_aspect("equal", adjustable="datalim")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._home_views: list[tuple] = []  # per-axes (xlim, ylim) full view
        self.toolbar = OverlayToolbar(self.canvas, root, self)

        bar = tk.Frame(root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        self.buttons = []
        for label, cmd in (
            ("Open BRD outline…", self.open_outline),
            ("Open PCB top…", lambda: self.open_photo(0)),
            ("Open PCB bot…", lambda: self.open_photo(1)),
            ("Open P&P…", self.open_pnp),
            ("Open BOM…", self.open_bom),
            ("Check presence", self.run_presence_check),
        ):
            b = tk.Button(bar, text=label, command=cmd)
            b.pack(side=tk.LEFT, padx=4)
            self.buttons.append(b)

        self.show_frames = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Component frames", variable=self.show_frames,
                       command=self._toggle_frames).pack(side=tk.LEFT, padx=(12, 0))

        tk.Label(bar, text="Photo opacity:").pack(side=tk.LEFT, padx=(16, 2))
        self.alpha = tk.DoubleVar(value=100.0)
        tk.Scale(bar, from_=0, to=100, orient=tk.HORIZONTAL, length=160,
                 variable=self.alpha, showvalue=False,
                 command=self._on_alpha).pack(side=tk.LEFT)

        tk.Button(bar, text="Save overlay PNG…",
                  command=self.save_png).pack(side=tk.RIGHT, padx=4)

        self.status = tk.Label(root, text="Open a board outline to begin.",
                               anchor="w", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        self._pan = None  # (axes, press x/y px, xlim, ylim) during right-drag
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Restore the previous session (CLI argument beats remembered path).
        dxf = dxf_override or self.settings.get("outline")
        if dxf and os.path.isfile(dxf):
            self.root.after(100, lambda: self.set_outline(dxf))
        pnp = self.settings.get("pnp")
        if pnp and os.path.isfile(pnp):
            self.root.after(200, lambda: self.set_pnp(pnp))
        bom = self.settings.get("bom")
        if bom and os.path.isfile(bom):
            self.root.after(300, lambda: self.set_bom(bom))

    # ---- file selection ----------------------------------------------------

    def _ask_open(self, title: str, filetypes, remember_key: str) -> str | None:
        prev = self.settings.get(remember_key)
        initdir = os.path.dirname(prev) if prev else None
        path = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                          initialdir=initdir)
        return path or None

    def open_outline(self):
        path = self._ask_open("Select board outline DXF", DXF_TYPES, "outline")
        if path:
            self.set_outline(path)

    def open_photo(self, side: int):
        if self.outline is None:
            messagebox.showinfo("No board outline",
                                "Open a board outline DXF first.")
            return
        key = SIDE_KEYS[side]
        path = self._ask_open(f"Photo of {SIDE_NAMES[side]}", IMAGE_TYPES, key)
        if path:
            self._start_jobs([(side, path)])

    # ---- outline -----------------------------------------------------------

    def set_outline(self, path: str):
        try:
            outline = load_outline(path)
        except Exception as ex:
            messagebox.showerror("Cannot load outline", f"{path}\n\n{ex}")
            return
        self.outline = outline
        self.settings["outline"] = path
        save_settings(self.settings)
        self.root.title(f"PCB Photo Overlay — {os.path.basename(path)}")

        mx = 0.05 * (outline.xmax - outline.xmin)
        my = 0.02 * (outline.ymax - outline.ymin)
        for i, (ax, name) in enumerate(zip(self.axes, SIDE_NAMES)):
            if self.images[i] is not None:
                self.images[i].remove()
                self.images[i] = None
            self.warped[i] = None
            self._clear_presence(i)
            if self.outline_artists[i] is not None:
                self.outline_artists[i].remove()
            sx = SIDE_XSIGN[i]
            segs = [s * (sx, 1.0) for s in outline.segments]
            self.outline_artists[i] = ax.add_collection(
                LineCollection(segs, colors="magenta",
                               linewidths=1.2, zorder=3))
            ax.set_title(name)
            xlims = sorted((sx * (outline.xmin - mx), sx * (outline.xmax + mx)))
            ax.set_xlim(*xlims)
            ax.set_ylim(outline.ymin - my, outline.ymax + my)
        self._home_views = [(ax.get_xlim(), ax.get_ylim()) for ax in self.axes]
        self.canvas.draw_idle()
        self.status.config(text=f"Outline loaded: {os.path.basename(path)}")

        # Re-register the remembered photos against the (new) outline.
        jobs = [(i, self.settings.get(k)) for i, k in enumerate(SIDE_KEYS)]
        jobs = [(i, p) for i, p in jobs if p and os.path.isfile(p)]
        if jobs:
            self._start_jobs(jobs)

    # ---- pick & place / component frames -------------------------------------

    def open_pnp(self):
        path = self._ask_open("Select pick & place file", PNP_TYPES, "pnp")
        if path:
            self.set_pnp(path)

    def set_pnp(self, path: str):
        try:
            self.parts = load_pnp(path)
        except Exception as ex:
            messagebox.showerror("Cannot load P&P", f"{path}\n\n{ex}")
            return
        self.settings["pnp"] = path
        save_settings(self.settings)
        for i in range(2):
            self._clear_presence(i)
        self._draw_frames()
        n_top = sum(1 for p in self.parts if p.side == "top")
        approx = sum(1 for p in self.parts if not p.size_exact)
        self.status.config(
            text=f"P&P loaded: {len(self.parts)} parts "
                 f"({n_top} top / {len(self.parts) - n_top} bottom), "
                 f"{approx} with approximate size (dashed frames)")

    def _draw_frames(self):
        for artists in self.frame_artists:
            for a in artists:
                a.remove()
        self.frame_artists = [[], []]
        if not self.parts:
            self.canvas.draw_idle()
            return
        for i, side in enumerate(SIDE_KEYS):
            ax = self.axes[i]
            exact, approx = [], []
            for p in self.parts:
                if p.side != side:
                    continue
                c = p.corners() * (SIDE_XSIGN[i], 1.0)
                (exact if p.size_exact else approx).append(
                    list(map(tuple, c)) + [tuple(c[0])])
            artists = []
            if exact:
                artists.append(ax.add_collection(LineCollection(
                    exact, colors="yellow", linewidths=0.7, zorder=4)))
            if approx:
                artists.append(ax.add_collection(LineCollection(
                    approx, colors="orange", linewidths=0.7,
                    linestyles="dashed", zorder=4)))
            for a in artists:
                a.set_visible(self.show_frames.get())
            self.frame_artists[i] = artists
        self.canvas.draw_idle()

    # ---- BOM -----------------------------------------------------------------

    def open_bom(self):
        path = self._ask_open("Select BOM", BOM_TYPES, "bom")
        if path:
            self.set_bom(path)

    def set_bom(self, path: str):
        try:
            self.bom = load_bom(path)
        except Exception as ex:
            messagebox.showerror("Cannot load BOM", f"{path}\n\n{ex}")
            return
        self.settings["bom"] = path
        save_settings(self.settings)
        n_dnp = sum(1 for b in self.bom.values() if b.dnp)
        self.status.config(
            text=f"BOM loaded: {len(self.bom)} positions, {n_dnp} DNP")
        if any(self.presence):
            self._draw_verdict_frames()  # re-color with BOM knowledge

    def _expected_present(self, refdes: str) -> bool:
        """True when the BOM says a part must be mounted here. Positions
        missing from the BOM count as DNP (nothing should be mounted)."""
        if self.bom is None:
            return True  # no BOM: every P&P position is expected populated
        b = self.bom.get(refdes)
        return b is not None and not b.dnp

    # ---- presence check ------------------------------------------------------

    def run_presence_check(self):
        """Flag not-mounted components (red frames). Uses the CNN classifier
        when golden/presence_cnn.pt and torch are available, otherwise the
        solder-mask color heuristic."""
        if not self.parts:
            messagebox.showinfo("No P&P", "Open a pick & place file first.")
            return
        if all(w is None for w in self.warped):
            messagebox.showinfo("No photo", "Register at least one photo first.")
            return
        for b in self.buttons:
            b.config(state=tk.DISABLED)
        threading.Thread(target=self._presence_worker, daemon=True).start()

    def _presence_worker(self):
        use_cnn = presence_cnn.available()
        method = "CNN" if use_cnn else "color"
        checked = 0
        fit_notes: list[str] = []
        for i, side in enumerate(SIDE_KEYS):
            if self.warped[i] is None or self.reg_info[i] is None:
                continue
            self.root.after(0, self.status.config, {
                "text": f"Presence ({method}): compensating lens distortion — "
                        f"{SIDE_NAMES[i]} …"})
            try:
                path, hom, small_w = self.reg_info[i]
                full = load_photo(path, max_dim=100000)
                scale = full.shape[1] / small_w
                # self-calibrate residual distortion on the components,
                # then work from the corrected board raster
                samples = distortion.measure_offsets(
                    full, self.outline, self.parts, side, hom, scale)
                field = distortion.fit_field(samples)
                if field is not None:
                    fit_notes.append(
                        f"{side} fit {field.rmse_before:.2f}→"
                        f"{field.rmse_after:.2f}mm")
                disp = distortion.warp_board_raster(
                    full, self.outline, hom, scale, self.outline.scale, field)
                self.root.after(0, self._apply_corrected_image, i, disp)
                if use_cnn:
                    raster = distortion.warp_board_raster(
                        full, self.outline, hom, scale, presence_cnn.PPMM,
                        field)
                    res = presence_cnn.classify_raster(
                        raster, presence_cnn.PPMM, self.outline, self.parts,
                        side)
                else:
                    res = check_presence(disp, self.outline, self.parts, side)
            except Exception as ex:
                self.root.after(0, messagebox.showerror,
                                "Presence check failed", str(ex))
                continue
            checked += len(res)
            self.root.after(0, self._presence_side_done, i, res)
        self.root.after(0, self._presence_finished, checked, method, fit_notes)

    def _apply_corrected_image(self, i: int, disp):
        """Swap the displayed photo for the distortion-corrected raster."""
        self.warped[i] = disp
        if self.images[i] is None:
            return
        img = disp if i == 0 else disp[:, ::-1]
        self.images[i].set_data(img)
        self.canvas.draw_idle()

    def _presence_side_done(self, i: int, res: dict[str, PresenceResult]):
        self.presence[i] = res
        self._draw_verdict_frames()

    def _verdicts(self):
        """(missing, extra) refdes lists: BOM expectation vs detection."""
        missing, extra = [], []
        for pres in self.presence:
            for r in pres.values():
                exp = self._expected_present(r.part.refdes)
                if exp and not r.present:
                    missing.append(r.part.refdes)
                elif not exp and r.present:
                    extra.append(r.part.refdes)
        return sorted(missing), sorted(extra)

    def _presence_finished(self, checked: int, method: str,
                           fit_notes: list[str] | None = None):
        for b in self.buttons:
            b.config(state=tk.NORMAL)
        missing, extra = self._verdicts()
        names = ", ".join(missing + [f"{r}!" for r in extra])
        if len(names) > 100:
            names = names[:97] + "…"
        fit = f"  [distortion {', '.join(fit_notes)}]" if fit_notes else ""
        bom_note = "" if self.bom is not None else "  (no BOM: DNP unknown)"
        self.status.config(
            text=f"Presence ({method}): {checked} checked — "
                 f"{len(missing)} missing, {len(extra)} unexpected"
                 + (f": {names}" if names else "") + bom_note + fit)

    def _draw_verdict_frames(self):
        """Green = board matches the BOM (mounted where expected, empty on
        DNP/unknown positions); red = mismatch (missing or unexpected)."""
        for artists in self.absent_artists:
            for a in artists:
                a.remove()
        self.absent_artists = [[], []]
        for i in range(2):
            ok, bad = [], []
            for r in self.presence[i].values():
                c = r.part.corners() * (SIDE_XSIGN[i], 1.0)
                seg = list(map(tuple, c)) + [tuple(c[0])]
                exp = self._expected_present(r.part.refdes)
                (ok if r.present == exp else bad).append(seg)
            artists = []
            if ok:
                artists.append(self.axes[i].add_collection(
                    LineCollection(ok, colors="lime", linewidths=0.9,
                                   zorder=5)))
            if bad:
                artists.append(self.axes[i].add_collection(
                    LineCollection(bad, colors="red", linewidths=1.8,
                                   zorder=6)))
            self.absent_artists[i] = artists
        self.canvas.draw_idle()

    def _clear_presence(self, side: int):
        if self.presence[side]:
            self.presence[side] = {}
            self._draw_verdict_frames()

    def _toggle_frames(self):
        vis = self.show_frames.get()
        for artists in self.frame_artists:
            for a in artists:
                a.set_visible(vis)
        self.canvas.draw_idle()

    def reset_view(self):
        if not self._home_views:
            return
        for ax, (xlim, ylim) in zip(self.axes, self._home_views):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        self.canvas.draw_idle()

    # ---- zoom (wheel) & pan (left-drag) --------------------------------------

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax not in tuple(self.axes) or event.xdata is None:
            return
        f = 1 / 1.25 if event.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata - (event.xdata - x0) * f,
                    event.xdata + (x1 - event.xdata) * f)
        ax.set_ylim(event.ydata - (event.ydata - y0) * f,
                    event.ydata + (y1 - event.ydata) * f)
        self.canvas.draw_idle()

    def _on_press(self, event):
        # left-drag pans, unless a toolbar mode (zoom/pan) is armed
        if (event.button == 1 and not self.toolbar.mode
                and event.inaxes in tuple(self.axes)):
            self._pan = (event.inaxes, event.x, event.y,
                         event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def _on_release(self, event):
        if event.button == 1:
            self._pan = None

    def _on_hover(self, event):
        if self._pan is not None:
            ax, px, py, xlim, ylim = self._pan
            bbox = ax.get_window_extent()
            dx = (event.x - px) * (xlim[1] - xlim[0]) / bbox.width
            dy = (event.y - py) * (ylim[1] - ylim[0]) / bbox.height
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()
            return
        if event.inaxes not in tuple(self.axes) or not self.parts:
            return
        i = list(self.axes).index(event.inaxes)
        side, x_mm = SIDE_KEYS[i], SIDE_XSIGN[i] * event.xdata
        best, best_d2 = None, 3.0 ** 2  # search radius 3 mm
        for p in self.parts:
            if p.side != side:
                continue
            d2 = (p.x_mm - x_mm) ** 2 + (p.y_mm - event.ydata) ** 2
            if d2 < best_d2:
                best, best_d2 = p, d2
        if best is not None:
            size = f"{best.length_mm:g}×{best.width_mm:g}mm"
            if not best.size_exact:
                size += " (approx)"
            extra = ""
            r = self.presence[i].get(best.refdes)
            if r is not None:
                exp = self._expected_present(best.refdes)
                if r.present == exp:
                    verdict = "OK mounted" if r.present else "OK empty"
                else:
                    verdict = "MISSING" if exp else "UNEXPECTED PART"
                extra = f"  [{verdict}, bare {r.bare_frac:.0%}]"
            bnote = ""
            if self.bom is not None:
                b = self.bom.get(best.refdes)
                if b is None:
                    bnote = "  (not in BOM)"
                elif b.dnp:
                    bnote = "  (DNP)"
            self.status.config(
                text=f"{best.refdes}  {best.footprint}  {size}  "
                     f"rot {best.rot_deg:g}°{extra}{bnote}  — {best.comment}")

    # ---- photo registration (background) ------------------------------------

    def _start_jobs(self, jobs: list[tuple[int, str]]):
        for b in self.buttons:
            b.config(state=tk.DISABLED)
        threading.Thread(target=self._worker, args=(jobs, self.outline),
                         daemon=True).start()

    def _worker(self, jobs: list[tuple[int, str]], outline: Outline):
        for side, path in jobs:
            self.root.after(
                0, self.status.config,
                {"text": f"Registering {os.path.basename(path)} …"})
            try:
                rgb = load_photo(path)
                res = register_photo(rgb, outline, side=SIDE_KEYS[side])
            except Exception as ex:
                self.root.after(0, self._register_failed, path, str(ex))
                continue
            self.root.after(0, self._register_done, side, path, res,
                            rgb.shape[1])
        self.root.after(0, self._jobs_finished)

    def _jobs_finished(self):
        for b in self.buttons:
            b.config(state=tk.NORMAL)

    def _register_failed(self, path: str, msg: str):
        self.status.config(text=f"FAILED: {os.path.basename(path)} — {msg}")
        messagebox.showerror("Registration failed",
                             f"{os.path.basename(path)}\n\n{msg}")

    def _register_done(self, side: int, path: str, res, photo_w: int):
        ax = self.axes[side]
        if self.images[side] is not None:
            self.images[side].remove()
        o = self.outline
        if side == 0:
            img, extent = res.warped_rgb, o.extent
        else:  # bottom view: flip back so the photo shows as shot
            img = res.warped_rgb[:, ::-1]
            extent = (-o.xmax, -o.xmin, o.ymin, o.ymax)
        self.images[side] = ax.imshow(
            img, extent=extent, zorder=1,
            alpha=self.alpha.get() / 100.0, interpolation="bilinear")
        self.warped[side] = res.warped_rgb
        self.reg_info[side] = (path, res.homography, photo_w)
        self._clear_presence(side)

        self.settings[SIDE_KEYS[side]] = path
        save_settings(self.settings)

        name = SIDE_NAMES[side]
        good = res.iou >= IOU_WARN
        ax.set_title(f"{name} — fit IoU {res.iou:.3f}"
                     + ("" if good else "  ⚠ check alignment"),
                     color="black" if good else "darkorange")
        self.status.config(
            text=f"{name}: {os.path.basename(path)} registered, IoU={res.iou:.3f}"
                 + ("" if good else " — LOW, verify the overlay"))
        self.canvas.draw_idle()

    # ---- misc ---------------------------------------------------------------

    def _on_alpha(self, _val):
        a = self.alpha.get() / 100.0
        for im in self.images:
            if im is not None:
                im.set_alpha(a)
        self.canvas.draw_idle()

    def save_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG image", "*.png")],
            initialfile="board_overlay.png")
        if path:
            self.fig.savefig(path, dpi=200)
            self.status.config(text=f"Saved {path}")


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    dxf = argv[0] if argv else None

    root = tk.Tk()
    OverlayApp(root, dxf_override=dxf)
    root.mainloop()


if __name__ == "__main__":
    main()
