"""Mouse navigation for embedded Matplotlib plots, independent of Tk widgets."""
import math


class PlotNavigation:
    def __init__(self, canvas, toolbar=None, on_double_click=None, on_reset=None):
        self.canvas = canvas
        self.toolbar = toolbar
        self.on_double_click = on_double_click
        self.on_reset = on_reset
        self.drag = None
        self.home = {}
        self.home_pending = True
        self.layout_engine = None
        self.connections = [canvas.mpl_connect(name, callback) for name, callback in [
            ('scroll_event', self.scroll), ('button_press_event', self.press),
            ('button_release_event', self.release), ('motion_notify_event', self.motion),
            ('key_press_event', self.key), ('draw_event', self.drawn),
            ('figure_leave_event', self.release), ('resize_event', self.release)]]

    def _freeze_layout(self):
        figure = self.canvas.figure
        engine = figure.get_layout_engine()
        if engine is not None and self.layout_engine is None:
            self.layout_engine = engine
            figure.set_layout_engine(None)

    def _restore_layout(self):
        if self.layout_engine is not None:
            self.canvas.figure.set_layout_engine(self.layout_engine)
            self.layout_engine = None
            return True
        return False

    def _toolbar_active(self):
        return bool(getattr(self.toolbar, 'mode', ''))

    def _push(self):
        if self.toolbar is not None:
            self.toolbar.push_current()

    def new_data(self):
        self._restore_layout()
        self.drag = None
        self.home_pending = True

    def drawn(self, event=None):
        if self.home_pending:
            self.home = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.canvas.figure.axes}
            self.home_pending = False

    def reset(self, ax=None):
        axes = [ax] if ax is not None else [a for a in self.canvas.figure.axes if a.get_visible()]
        self._push()
        for item in axes:
            if item in self.home:
                xlim, ylim = self.home[item]
                item.set_xlim(xlim); item.set_ylim(ylim)
            else:
                item.autoscale(enable=True)
        if self.on_reset:
            self.on_reset()
        self._push()
        self.canvas.draw_idle()

    def scroll(self, event):
        ax = event.inaxes
        if ax is None or not ax.get_visible() or self._toolbar_active():
            return
        if event.xdata is None or event.ydata is None:
            return
        step = getattr(event, 'step', 0)
        if not step:
            step = 1 if event.button == 'up' else -1
        scale = 1.25 ** (-max(-5, min(5, step)))
        key = (getattr(event, 'key', '') or '').lower()
        zoom_x = 'control' not in key and 'ctrl' not in key
        zoom_y = 'shift' not in key
        self._push()
        for enabled, anchor, limits, setter in [
            (zoom_x, event.xdata, ax.get_xlim(), ax.set_xlim),
            (zoom_y, event.ydata, ax.get_ylim(), ax.set_ylim)]:
            if enabled:
                new = tuple(anchor+(v-anchor)*scale for v in limits)
                if all(math.isfinite(v) for v in new) and abs(new[1]-new[0]) > 1e-12:
                    setter(new)
        self._push()
        self.canvas.draw_idle()

    def press(self, event):
        ax = event.inaxes
        if ax is None or not ax.get_visible():
            return
        if getattr(event, 'dblclick', False):
            if self._toolbar_active():
                return
            self._restore_layout()
            self.drag = None
            if self.on_double_click:
                self.on_double_click(ax)
            return
        if event.button not in (1, 2) or event.xdata is None or event.ydata is None:
            return
        self._freeze_layout()
        if self._toolbar_active():
            return
        self._push()
        self.drag = (ax, ax.transData.frozen().inverted(), event.xdata, event.ydata,
                     ax.get_xlim(), ax.get_ylim())

    def motion(self, event):
        if self.drag is None:
            return
        ax, transform, x, y, xlim, ylim = self.drag
        if event.x is None or event.y is None:
            return
        current = transform.transform((event.x, event.y))
        dx, dy = current[0]-x, current[1]-y
        ax.set_xlim(xlim[0]-dx, xlim[1]-dx)
        ax.set_ylim(ylim[0]-dy, ylim[1]-dy)
        self.canvas.draw_idle()

    def release(self, event=None):
        changed = self.drag is not None
        if self.drag is not None:
            self.drag = None
            self._push()
        if self._restore_layout() or changed:
            self.canvas.draw_idle()

    def key(self, event):
        if (event.key or '').lower() == 'r':
            self.reset(event.inaxes)

    def close(self):
        self._restore_layout()
        for connection in self.connections:
            self.canvas.mpl_disconnect(connection)
        self.connections.clear()
        self.drag = None
