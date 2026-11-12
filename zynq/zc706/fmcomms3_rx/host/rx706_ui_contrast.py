"""Readable amber hints on both light and dark native ttk surfaces."""
from tkinter import ttk


def luminance(rgb):
    channels = [v / 65535 for v in rgb]
    channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    return sum(v * weight for v, weight in zip(channels, (.2126, .7152, .0722)))


def contrast(a, b):
    x, y = sorted((luminance(a), luminance(b)))
    return (y + .05) / (x + .05)


def bright_hint(label):
    """Resolve native colors through Tk; re-evaluate after theme changes."""
    def refresh(event=None):
        style = ttk.Style(label)
        background = style.lookup('TFrame', 'background') or label.winfo_toplevel().cget('background')
        rgb = label.winfo_rgb(background)
        candidates = ('#ffe08a', '#704800', '#ffffff', '#000000')
        color = next((c for c in candidates if contrast(label.winfo_rgb(c), rgb) >= 4.5),
                     max(candidates, key=lambda c: contrast(label.winfo_rgb(c), rgb)))
        label.configure(foreground=color, background=background)
    refresh()
    label.bind('<<ThemeChanged>>', refresh, add='+')
    return label
