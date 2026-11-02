"""Stable, DPI-aware widths for horizontally scrollable PWM symbol tables."""
from tkinter import font as tkfont, ttk


def configure_symbol_table(tree, columns, headers, total_chips, code_chips):
    body = tkfont.nametofont('TkDefaultFont', root=tree)
    heading = tkfont.nametofont('TkHeadingFont', root=tree)
    padding = max(14, body.measure('00'))
    style = ttk.Style(tree)
    style.configure('RxPWMSymbols.Treeview', font=body,
                    rowheight=body.metrics('linespace')+max(6, padding//2))
    style.configure('RxPWMSymbols.Treeview.Heading', font=heading)
    tree.configure(style='RxPWMSymbols.Treeview', displaycolumns=columns)
    widths = {}
    for key, title in zip(columns, headers):
        minimum = max(heading.measure(title)+padding, body.measure('0000.0000')+padding)
        if key == 'index':
            minimum = max(heading.measure(title)+padding, body.measure('000000')+padding)
        if key == 'pattern':
            preferred = min(body.measure('1'*total_chips)+padding, body.measure('0'*60)+padding)
        elif key in ('bits', 'candidate_bits', 'expected_bits'):
            preferred = min(body.measure('1'*code_chips)+padding, body.measure('0'*60)+padding)
        elif key in ('reason', 'raw_reason', 'candidate_source', 'warning'):
            preferred = body.measure('状态与判决说明'*4)+padding
        else:
            preferred = minimum
        widths[key] = (max(minimum, preferred), minimum)
        tree.heading(key, text=title, anchor='center')

    def restore():
        for key, (width, minimum) in widths.items():
            tree.column(key, width=width, minwidth=minimum, stretch=False,
                        anchor='w' if key in ('pattern', 'bits', 'candidate_bits', 'expected_bits',
                                              'reason', 'raw_reason', 'warning', 'candidate_source') else 'center')
        tree.xview_moveto(0)

    def clamp(event=None):
        # One native Treeview owns headings and rows. Never shrink all columns
        # to the viewport on Configure; retain the horizontal scroll range.
        for key, (_, minimum) in widths.items():
            if int(tree.column(key, 'width')) < minimum:
                tree.column(key, width=minimum)

    restore()
    tree.bind('<ButtonRelease-1>', clamp, add='+')
    tree.bind('<Configure>', clamp, add='+')
    return restore
