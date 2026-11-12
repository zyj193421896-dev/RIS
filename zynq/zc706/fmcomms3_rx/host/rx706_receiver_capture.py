"""Board acquisition inside the receiver app, using the original controller.

Opening the receiver or this panel never connects or captures automatically.
A completed, verified download is loaded only by an explicit user action.
"""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import rx706 as protocol
from rx706_gui import Rx706App


class ReceiverCaptureApp(Rx706App):
    """Keep all existing acquisition controls; return to the owning receiver."""

    def __init__(self, root, bridge):
        self.bridge = bridge
        super().__init__(root)
        root.title('RIS 收端 · 板卡连接与采集')
        root.protocol('WM_DELETE_WINDOW', self.hide)
        self.spectrum_button.configure(text='载入最近采集 → 验收', command=self.open_spectrum)
        self.notice.set('先连接板卡，再采集 / 下载。下载完成后点击“载入最近采集 → 验收”；不会自动解调。')

    def hide(self):
        # Hiding this window must not cancel an in-flight transfer or release DDR.
        self.root.withdraw()
        self.bridge.refresh_status()

    def open_spectrum(self):
        self.bridge.load_latest()
        self.bridge.view.window.deiconify()
        self.bridge.view.window.lift()

    def handle_event(self, kind, value):
        super().handle_event(kind, value)
        if kind == 'saved':
            self.bridge.received(*value)
        elif kind in ('connected', 'status', 'phase', 'progress', 'error', 'notice', 'done'):
            self.bridge.refresh_status()


class ReceiverCaptureBridge:
    """Shared acquisition entry for all three receiver workflows.

    It owns one optional child window/controller. It deliberately does not use
    SpectrumWindow.offer_download: that legacy path may auto-run analysis.
    """

    def __init__(self, view):
        self.view = view
        self.app = None
        self.closed = False
        self.latest = None
        self.status = tk.StringVar(master=view.window, value='板卡：未连接；离线 BIN 仍可直接打开')
        self.latest_text = tk.StringVar(master=view.window, value='尚无本次会话完成的采集下载')
        self.panel = ttk.LabelFrame(view.outer, text='板卡数据采集 · 三功能区共用', padding=(8, 5))
        before = view.outer.winfo_children()[0]
        self.panel.pack(fill='x', before=before, pady=(0, 6))
        row = ttk.Frame(self.panel)
        row.pack(fill='x')
        self.open_button = ttk.Button(row, text='连接板卡 / 采集…', command=self.show)
        self.open_button.pack(side='left', padx=(0, 10))
        self.load_button = ttk.Button(row, text='载入最近采集 → 当前功能区', command=self.load_latest,
                                      state='disabled')
        self.load_button.pack(side='left', padx=(0, 10))
        ttk.Label(row, textvariable=self.status).pack(side='left', fill='x', expand=True)
        ttk.Label(self.panel, textvariable=self.latest_text, wraplength=1100).pack(fill='x', pady=(3, 0))

    def show(self):
        if self.closed or self.view.closed:
            return None
        if self.app is None:
            child = tk.Toplevel(self.view.window)
            child.withdraw()
            try:
                self.app = ReceiverCaptureApp(child, self)
            except Exception:
                child.destroy()
                raise
        self.app.root.deiconify()
        self.app.root.lift()
        self.refresh_status()
        return self.app

    def refresh_status(self):
        if self.closed:
            return
        app = self.app
        if app is None:
            self.status.set('板卡：未连接；离线 BIN 仍可直接打开')
            return
        state = {'empty': '可采集', 'capturing': '采集中', 'ready': 'DDR 数据就绪',
                 'failed': '采集失败', 'unavailable': '等待启动检查'}.get(app.info.get('state'), '状态未知')
        operation = {'connect': '连接中', 'capture': '采集中', 'download': '下载中',
                     'status': '刷新状态中', 'release': '释放中', 'key': '业务操作中'}.get(app.operation, '')
        connection = '已连接' if app.connected else '未连接'
        self.status.set(f'板卡：{connection}；{operation or (state if app.connected else "可继续离线验收")}')

    def received(self, filename, metadata):
        if self.closed:
            return
        self.latest = (str(filename), metadata)
        self.latest_text.set(f'已校验保存：{filename}；点击“载入最近采集”选入当前功能区，不自动运行。')
        self.load_button.configure(state='normal')
        self.refresh_status()

    def load_latest(self):
        view = self.view
        if self.closed or view.closed:
            return False
        if self.latest is None:
            view.status.set('尚无本次会话下载完成的采集；也可用“打开 bin…”选择已有文件。')
            return False
        if view.busy or view.workflows.editor_open():
            view.status.set('接收运行中或配置编辑窗口未关闭；请结束后再载入最近采集。')
            return False
        filename, metadata = self.latest
        path = Path(filename)
        try:
            size = protocol.validate_capture(metadata['capture'])
            if (not metadata.get('complete') or metadata.get('file_bytes') != size
                    or not path.is_file() or path.stat().st_size != size
                    or not Path(str(path) + '.json').is_file()):
                raise ValueError('完整的 BIN / JSON 文件不存在或长度发生变化，请检查保存路径。')
            saved = json.loads(Path(str(path) + '.json').read_text(encoding='utf-8'))
            if (not isinstance(saved, dict) or saved.get('complete') is not True
                    or saved.get('capture') != metadata['capture']
                    or saved.get('file_bytes') != size or saved.get('sha256') != metadata.get('sha256')):
                raise ValueError('同名 JSON 已改变或不属于本次已验证下载，请检查文件配对。')
        except (OSError, ValueError, KeyError, TypeError, protocol.ProtocolError) as exc:
            view.status.set('采集文件暂不能载入：' + str(exc))
            return False
        # A new capture is never evidence of the sender's current configuration.
        # Keep all parameters, but require a fresh explicit confirmation.
        view.workflows.confirmed = None
        view.filename.set(filename)
        view.workflows.refresh()
        view.status.set(f'已载入新采集到 {view.workflows.area}；请确认本次发送配置，再点击“运行接收”。')
        self.latest_text.set(f'已载入 {view.workflows.area}：{filename}；BIN + JSON 保留在原路径。')
        return True

    def close(self):
        """Disconnect and retire Tk callbacks without issuing ABORT or RELEASE."""
        if self.closed:
            return
        self.closed = True
        app = self.app
        if app is None:
            return
        app.closing = True
        for callback in (app.poll_after, app.close_after):
            if callback:
                try:
                    app.root.after_cancel(callback)
                except tk.TclError:
                    pass
        app.poll_after = app.close_after = None
        app.controller.close()
        try:
            app.root.destroy()
        except tk.TclError:
            pass
