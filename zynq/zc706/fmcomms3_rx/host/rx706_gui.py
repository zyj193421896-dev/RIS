#!/usr/bin/env python3
"""Tk control panel; optional local FFT viewer uses NumPy and Matplotlib."""
import codecs
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
import math
from pathlib import Path
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import rx706 as protocol

DATA_RATE = 15360000 * 8


def capture_length(value, unit):
    """Round upward to a complete AXIS packet; return actual byte count."""
    try:
        amount = Decimal(str(value).strip())
        if not amount.is_finite() or amount <= 0:
            raise ValueError("采集长度必须是大于零的有限数值")
        if unit not in ("MiB", "秒"):
            raise ValueError("未知长度单位")
        scale = 1048576 if unit == "MiB" else DATA_RATE
        if amount > Decimal(protocol.MAX_BYTES) / scale:
            raise ValueError(f"最大为 {protocol.MAX_BYTES // 1048576} MiB，即约 {protocol.MAX_BYTES / DATA_RATE:.6f} 秒")
        raw = amount * scale
        packets = (raw / protocol.PACKET_BYTES).to_integral_value(rounding=ROUND_CEILING)
        return max(1, int(packets)) * protocol.PACKET_BYTES
    except InvalidOperation as exc:
        raise ValueError("请输入有效的采集长度") from exc


@dataclass(frozen=True)
class ConnectionSettings:
    host: str
    port: int = 5001
    console_port: int = 5000
    timeout: float = 30.0


class Disconnected(RuntimeError):
    pass


class Controller:
    """One worker owns the control socket; the console has its own reader.

    Workers communicate through queues/events and never call Tk. Even abort
    uses the existing control connection, respecting the single-client server.
    """
    def __init__(self):
        self.events = queue.Queue()
        self.actions = queue.Queue()
        self.cancel_event = threading.Event()
        self.disconnect_event = threading.Event()
        self.quit_event = threading.Event()
        self.lock = threading.Lock()
        self.client = None
        self.console_socket = None
        self.console_generation = 0
        self.settings = None
        self.last_status = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="rx706-control")
        self.thread.start()

    def emit(self, kind, value=None):
        self.events.put((kind, value))

    def submit(self, action, **options):
        if action in ("connect", "capture", "download"):
            self.cancel_event.clear()
        self.actions.put((action, options))

    def stop(self):
        self.cancel_event.set()

    def disconnect(self):
        self.disconnect_event.set()
        self.cancel_event.set()
        self.actions.put(("disconnect", {}))
        # Wake a blocked socket read without closing its buffered file in Tk.
        with self.lock:
            sockets = [self.client.socket if self.client else None, self.console_socket]
        threading.Thread(target=self._interrupt, args=(sockets,), daemon=True).start()

    def close(self):
        self.quit_event.set()
        self.disconnect()

    def _interrupt(self, sockets):
        for connection in sockets:
            if connection:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _close_connections(self):
        with self.lock:
            client, console = self.client, self.console_socket
            self.client = self.console_socket = None
            self.console_generation += 1
        if console:
            try:
                console.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            console.close()
        if client:
            client.close()
        self.emit("connected", False)
        self.emit("console", "未连接")

    def _check_connection(self):
        if self.disconnect_event.is_set() or self.quit_event.is_set():
            raise Disconnected("连接已断开，板上采集和已保存数据保留")
        if self.client is None:
            raise ValueError("请先连接板卡")

    def _console_reader(self, settings, generation):
        connection = None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            connection = socket.create_connection((settings.host, settings.console_port), settings.timeout)
            connection.settimeout(0.5)
            with self.lock:
                if generation != self.console_generation or self.disconnect_event.is_set():
                    return
                self.console_socket = connection
            self.emit("console", "已连接")
            self.emit("log", "\n--- TCP 控制台已连接，回放最近的板卡日志 ---\n")
            while not self.quit_event.is_set() and generation == self.console_generation:
                try:
                    data = connection.recv(8192)
                except socket.timeout:
                    continue
                if not data:
                    break
                self.emit("log", decoder.decode(data))
        except OSError as exc:
            if generation == self.console_generation and not self.disconnect_event.is_set():
                self.emit("log", f"\n日志连接失败：{exc}\n")
        finally:
            if connection:
                connection.close()
            with self.lock:
                current = generation == self.console_generation
                if current:
                    self.console_socket = None
            if current:
                self.emit("console", "已断开")

    def _connect(self, settings):
        self._close_connections()
        self.disconnect_event.clear()
        self.settings = settings
        self.emit("phase", "connect")
        client = protocol.Client(settings.host, settings.port, min(settings.timeout, 5))
        client.socket.settimeout(settings.timeout)
        with self.lock:
            self.client = client
        self._check_connection()
        info = client.request("HELLO")
        self.emit("status", info)
        self.emit("connected", True)
        self.emit("log", f"\n控制接口已连接：{settings.host}:{settings.port}\n")
        generation = self.console_generation
        threading.Thread(target=self._console_reader, args=(settings, generation),
                         daemon=True, name="rx706-console").start()

    def _status(self):
        self._check_connection()
        info = self.client.request("STATUS")
        self.last_status = time.monotonic()
        self.emit("status", info)
        return info

    def _capture(self, options):
        self._check_connection()
        self.emit("phase", "capture")
        self.client.request(f"CAPTURE {options['bytes']}")
        deadline = time.monotonic() + 90
        while True:
            self._check_connection()
            if self.cancel_event.is_set():
                self.client.request("ABORT")
                info = self._status()
                message = ("采集已完成，取消后续下载；数据保留在 DDR" if info["state"] == "ready"
                           else "采集已终止，该次部分数据无效")
                raise protocol.TransferCancelled(message)
            info = self._status()
            self.emit("progress", (info.get("received_bytes", 0), options["bytes"], 0, "capture"))
            if info["state"] != "capturing":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("采集等待超时；重连后查询状态，或显式终止采集")
            self.cancel_event.wait(0.2)
        protocol.validate_capture(info)
        self.emit("log", f"\n采集就绪：{info['capture_id']}，{info['bytes'] / 1048576:g} MiB\n")
        if options["auto_download"]:
            self._download(options)
        else:
            self.emit("notice", "采集完成，数据保留在 DDR，可点击下载")

    def _download(self, options):
        self._check_connection()
        self.emit("phase", "download")

        def progress(done, total, rate, phase):
            self.emit("progress", (done, total, rate, phase))

        saved = protocol.download(self.client, options["output"],
                                  resume=options.get("resume", False), note=options.get("note", ""),
                                  progress=False, progress_callback=progress, cancel_event=self.cancel_event)
        self.emit("saved", (str(options["output"]), saved))
        self._status()

    def _run_action(self, action, options):
        if action == "connect":
            self._connect(options["settings"])
        elif action == "disconnect":
            self._close_connections()
            self.emit("notice", "已断开连接，板上采集和数据保留")
        elif action == "status":
            self._status()
        elif action == "capture":
            self._capture(options)
        elif action == "download":
            self._download(options)
        elif action == "key":
            self._check_connection()
            self.client.request(f"KEY {options['key']}")
            self.emit("log", f"\n>>> 已发送业务按键：{options['key']}\n")
        elif action == "release":
            self._check_connection()
            self.client.request(f"RELEASE {options['capture_id']}")
            self._status()
            self.emit("notice", "DDR 数据已释放，可以开始下一次采集")

    def _run(self):
        try:
            while not self.quit_event.is_set():
                try:
                    action, options = self.actions.get(timeout=0.2)
                except queue.Empty:
                    if self.client and not self.disconnect_event.is_set() and time.monotonic() - self.last_status > 2:
                        try:
                            self._status()
                        except Exception as exc:
                            self._close_connections()
                            self.emit("error", str(exc))
                    continue
                try:
                    self._run_action(action, options)
                except protocol.TransferCancelled as exc:
                    self.emit("notice", str(exc))
                except Disconnected as exc:
                    self._close_connections()
                    self.emit("notice", str(exc))
                except Exception as exc:
                    if action == "connect" or isinstance(exc, (OSError, protocol.ProtocolError)):
                        self._close_connections()
                    self.emit("error", "连接已断开，数据可在重连后检查" if self.disconnect_event.is_set() else str(exc))
                finally:
                    self.emit("done", action)
        finally:
            self._close_connections()


class Rx706App:
    def __init__(self, root):
        self.root = root
        self.controller = Controller()
        self.connected = self.busy = self.closing = False
        self.info = {}
        self.operation = ""
        self.editable = []
        self.connection_entries = []
        self.key_buttons = []
        self.spectrum_window = None
        self.last_download = ""
        root.title("ZC706 · FMCOMMS3 IQ 采集")
        root.geometry("1080x940")
        root.minsize(1020, 800)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.host = tk.StringVar(value="192.168.1.10")
        self.port = tk.StringVar(value="5001")
        self.console_port = tk.StringVar(value="5000")
        self.timeout = tk.StringVar(value="30")
        self.length_unit = tk.StringVar(value="秒")
        self.length_value = tk.StringVar(value="1")
        self.output = tk.StringVar(value=self.default_path())
        self.note = tk.StringVar()
        self.auto_download = tk.BooleanVar(value=True)
        self.length_hint = tk.StringVar()
        self.connection_text = tk.StringVar(value="控制：未连接　日志：未连接")
        self.board_text = tk.StringVar(value="板卡状态：未知")
        self.notice = tk.StringVar(value="将电脑有线网卡设为 192.168.1.20/24，连接板卡后点击连接")
        self.progress_text = tk.StringVar(value="等待操作")
        self.console_state = "未连接"
        self._build()
        self.length_value.trace_add("write", self.update_length)
        self.length_unit.trace_add("write", self.update_length)
        self.update_length()
        self.update_buttons()
        self.close_after = None
        self.poll_after = root.after(50, self.poll_events)

    @staticmethod
    def default_path():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return str(Path(__file__).resolve().parent / 'gui_data' / 'current_pwm' / f"rx_{stamp}.bin")

    def _build(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 17, "bold"), foreground="#16304c")
        style.configure("Accent.TButton", foreground="#ffffff", background="#2467a4", padding=(12, 6))
        style.map("Accent.TButton", background=[("active", "#1d5487"), ("disabled", "#a5b4c3")])
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)
        ttk.Label(outer, text="ZC706 · 双通道 IQ 采集", style="Title.TLabel").grid(row=0, sticky="w")
        ttk.Label(outer, text="15.36 MSPS / I1 Q1 I2 Q2 / 小端 int16 / 最大 2044 MiB / 约 17.442 秒").grid(row=1, sticky="w", pady=(3, 10))

        connection = ttk.LabelFrame(outer, text="网络连接", padding=9)
        connection.grid(row=2, sticky="ew", pady=(0, 8))
        for col, (label, variable, width) in enumerate([
            ("板卡 IP", self.host, 19), ("控制端口", self.port, 7),
            ("日志端口", self.console_port, 7), ("超时(s)", self.timeout, 5)]):
            ttk.Label(connection, text=label).grid(row=0, column=col*2, padx=(0, 4))
            entry = ttk.Entry(connection, textvariable=variable, width=width)
            entry.grid(row=0, column=col*2+1, padx=(0, 12), sticky="ew")
            self.connection_entries.append(entry)
        self.connect_button = ttk.Button(connection, text="连接", command=self.connect)
        self.connect_button.grid(row=0, column=8, padx=3)
        self.disconnect_button = ttk.Button(connection, text="断开", command=self.controller.disconnect)
        self.disconnect_button.grid(row=0, column=9, padx=3)
        self.status_button = ttk.Button(connection, text="刷新状态", command=lambda: self.start("status"))
        self.status_button.grid(row=0, column=10, padx=3)
        ttk.Label(connection, textvariable=self.connection_text).grid(row=1, column=0, columnspan=11, sticky="w", pady=(7, 0))
        ttk.Label(connection, textvariable=self.board_text).grid(row=2, column=0, columnspan=11, sticky="w", pady=(3, 0))

        capture = ttk.LabelFrame(outer, text="采集与文件", padding=9)
        capture.grid(row=3, sticky="ew", pady=(0, 8))
        capture.columnconfigure(1, weight=1)
        length = ttk.Frame(capture)
        length.grid(row=0, column=0, columnspan=4, sticky="ew")
        ttk.Label(length, text="采集长度").pack(side="left", padx=(0, 8))
        mode = ttk.Combobox(length, textvariable=self.length_unit, values=("MiB", "秒"), state="readonly", width=6)
        mode.pack(side="left", padx=(0, 5))
        amount = ttk.Entry(length, textvariable=self.length_value, width=13)
        amount.pack(side="left", padx=(0, 12))
        self.editable.extend([mode, amount])
        for text, value, unit in [("128 KiB", "0.125", "MiB"), ("8 MiB", "8", "MiB"),
                                  ("1 秒", "1", "秒"), ("满容量", "2044", "MiB")]:
            button = ttk.Button(length, text=text, command=lambda v=value, u=unit: self.preset(v, u))
            button.pack(side="left", padx=3)
            self.editable.append(button)
        ttk.Label(capture, textvariable=self.length_hint).grid(row=1, column=0, columnspan=4, sticky="w", pady=(7, 8))
        ttk.Label(capture, text="保存路径").grid(row=2, column=0, sticky="w", padx=(0, 8))
        path_entry = ttk.Entry(capture, textvariable=self.output)
        path_entry.grid(row=2, column=1, sticky="ew", padx=(0, 6))
        browse = ttk.Button(capture, text="选择…", command=self.browse)
        browse.grid(row=2, column=2, padx=3)
        fresh = ttk.Button(capture, text="新文件名", command=lambda: self.output.set(self.default_path()))
        fresh.grid(row=2, column=3, padx=3)
        self.editable.extend([path_entry, browse, fresh])
        ttk.Label(capture, text="实验备注").grid(row=3, column=0, sticky="w", pady=8)
        note_entry = ttk.Entry(capture, textvariable=self.note)
        note_entry.grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=8)
        check = ttk.Checkbutton(capture, text="采集完成后自动下载", variable=self.auto_download)
        check.grid(row=3, column=2, columnspan=2, sticky="w")
        self.editable.extend([note_entry, check])
        actions = ttk.Frame(capture)
        actions.grid(row=4, column=0, columnspan=4, sticky="ew")
        self.capture_button = ttk.Button(actions, text="开始采集", style="Accent.TButton", command=self.capture)
        self.download_button = ttk.Button(actions, text="下载当前数据", command=lambda: self.download(False))
        self.resume_button = ttk.Button(actions, text="断点续传", command=lambda: self.download(True))
        self.stop_button = ttk.Button(actions, text="停止当前操作", command=self.controller.stop)
        self.release_button = ttk.Button(actions, text="释放 DDR 数据", command=self.release)
        for button in [self.capture_button, self.download_button, self.resume_button, self.stop_button, self.release_button]:
            button.pack(side="left", padx=(0, 9))
        self.spectrum_button = ttk.Button(actions, text="IQ 信号分析…", command=self.open_spectrum)
        self.spectrum_button.pack(side="left", padx=(0, 9))

        transfer = ttk.Frame(outer)
        transfer.grid(row=4, sticky="ew", pady=(0, 8))
        transfer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(transfer, maximum=100)
        self.progress.grid(row=0, sticky="ew")
        ttk.Label(transfer, textvariable=self.progress_text).grid(row=1, sticky="w", pady=4)

        keys = ttk.LabelFrame(outer, text="原串口业务 · TCP 控制台", padding=8)
        keys.grid(row=5, sticky="ew", pady=(0, 8))
        for col, (key, text) in enumerate([("g", "增益 g"), ("s", "快照 s"), ("0", "确认 RF关 0"),
                                         ("1", "确认916M 1"), ("2", "确认917M 2"),
                                         ("q", "退出测试 q"), ("t", "重新测试 t")]):
            button = ttk.Button(keys, text=text, command=lambda k=key: self.start("key", key=k))
            button.grid(row=0, column=col, padx=(0, 6), sticky="ew")
            keys.columnconfigure(col, weight=1)
            self.key_buttons.append(button)
        ttk.Label(keys, text="0/1/2 仅确认程序步骤，矢网频率、功率和 RF 开关仍需手动设置。").grid(row=1, column=0, columnspan=7, sticky="w", pady=(7, 0))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=6, sticky="nsew")
        log_tab, status_tab = ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(log_tab, text="板卡日志")
        notebook.add(status_tab, text="状态详情")
        tools = ttk.Frame(log_tab)
        tools.pack(fill="x", pady=5)
        ttk.Button(tools, text="清空显示", command=lambda: self.log_text.delete("1.0", "end")).pack(side="left", padx=4)
        ttk.Button(tools, text="保存日志…", command=self.save_log).pack(side="left", padx=4)
        self.log_text = scrolledtext.ScrolledText(log_tab, height=10, wrap="word", font="TkFixedFont",
                                                 background="#101b2d", foreground="#dce6f3", insertbackground="white")
        self.log_text.pack(fill="both", expand=True)
        self.detail_text = scrolledtext.ScrolledText(status_tab, wrap="none", height=10, font="TkFixedFont", state="disabled")
        self.detail_text.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.notice, wraplength=1020, foreground="#24486c").grid(row=7, sticky="ew", pady=(8, 0))

    def preset(self, value, unit):
        self.length_unit.set(unit)
        self.length_value.set(value)

    def update_length(self, *_):
        try:
            count = capture_length(self.length_value.get(), self.length_unit.get())
            seconds = count / DATA_RATE
            span = f"{seconds:.6g} 秒" if seconds >= 1 else f"{seconds * 1000:.6g} ms"
            self.length_hint.set(f"实际：{count / 1048576:g} MiB  /  每通道 {count // 8:,} 点  /  约 {span}（按 32 KiB 向上对齐）")
        except ValueError as exc:
            self.length_hint.set(str(exc))

    def start(self, action, **options):
        if self.busy:
            return
        self.busy, self.operation = True, action
        self.notice.set({"capture": "正在准备采集…", "download": "正在准备下载…",
                         "connect": "正在连接板卡…"}.get(action, "正在处理…"))
        self.update_buttons()
        self.controller.submit(action, **options)

    def connect(self):
        try:
            settings = ConnectionSettings(self.host.get().strip(), int(self.port.get()),
                                          int(self.console_port.get()), float(self.timeout.get()))
            if (not settings.host or not 1 <= settings.port <= 65535
                    or not 1 <= settings.console_port <= 65535
                    or not math.isfinite(settings.timeout) or not 1 <= settings.timeout <= 120):
                raise ValueError("请检查 IP、端口及超时（1～120 秒）")
            self.start("connect", settings=settings)
        except ValueError as exc:
            messagebox.showerror("连接参数", str(exc), parent=self.root)

    def output_options(self, resume=False):
        value = self.output.get().strip()
        if not value:
            raise ValueError("请选择保存文件路径")
        output = Path(value).expanduser()
        if output.name.endswith(".part"):
            output = output.with_name(output.name[:-5])
            self.output.set(str(output))
        if output.exists() or Path(str(output) + ".json").exists():
            raise ValueError("输出文件已存在，请选择新文件名")
        partial, checkpoint = Path(str(output) + ".part"), Path(str(output) + ".part.json")
        if resume and not (partial.is_file() and checkpoint.is_file()):
            raise ValueError("此路径没有完整的 .part / .part.json，请选择原下载文件路径")
        if not resume and (partial.exists() or checkpoint.exists()):
            raise ValueError("此路径已有部分文件，请使用断点续传，或选择新文件名")
        return {"output": output, "note": self.note.get(), "resume": resume}

    def capture(self):
        try:
            count = capture_length(self.length_value.get(), self.length_unit.get())
            auto = self.auto_download.get()
            options = self.output_options() if auto else {}
            self.start("capture", bytes=count, auto_download=auto, **options)
        except ValueError as exc:
            messagebox.showerror("采集参数", str(exc), parent=self.root)

    def download(self, resume):
        try:
            self.start("download", **self.output_options(resume))
        except ValueError as exc:
            messagebox.showerror("保存路径", str(exc), parent=self.root)

    def release(self):
        if self.info.get("state") == "ready":
            self.start("release", capture_id=self.info["capture_id"])

    def browse(self):
        path = Path(self.output.get()).expanduser()
        selected = filedialog.asksaveasfilename(parent=self.root, title="选择原始 IQ 文件路径（续传时选择原文件名）",
                     initialdir=str(path.parent if path.parent.is_dir() else Path.cwd()), initialfile=path.name,
                     defaultextension=".bin", filetypes=[("原始 IQ", "*.bin"), ("所有文件", "*")], confirmoverwrite=False)
        if selected:
            self.output.set(selected)

    def save_log(self):
        name = filedialog.asksaveasfilename(parent=self.root, defaultextension=".txt", filetypes=[("日志", "*.txt")])
        if name:
            try:
                Path(name).write_text(self.log_text.get("1.0", "end-1c"), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("保存日志", str(exc), parent=self.root)

    def open_spectrum(self):
        try:
            if self.spectrum_window is None or self.spectrum_window.closed:
                from rx706_spectrum_gui import SpectrumWindow
                filename = self.last_download or self.output.get().strip()
                self.spectrum_window = SpectrumWindow(self.root, filename)
            self.spectrum_window.window.deiconify()
            self.spectrum_window.window.lift()
        except ImportError as exc:
            messagebox.showerror("频谱依赖", "频谱分析需要 NumPy 和 Matplotlib。\n"
                "可安装：python3 -m pip install numpy matplotlib\n" + str(exc), parent=self.root)

    def update_buttons(self):
        idle = self.connected and not self.busy
        state = self.info.get("state")
        self.connect_button.configure(state="normal" if not self.connected and not self.busy else "disabled")
        self.disconnect_button.configure(state="normal" if self.connected or self.busy else "disabled")
        self.status_button.configure(state="normal" if idle else "disabled")
        can_capture = state == "empty" or (state == "failed" and self.info.get("reset_verified") == 1)
        self.capture_button.configure(state="normal" if idle and can_capture else "disabled")
        for button in (self.download_button, self.resume_button, self.release_button):
            button.configure(state="normal" if idle and state == "ready" else "disabled")
        self.stop_button.configure(state="normal" if self.busy and self.operation in ("capture", "download") else "disabled")
        for button in self.key_buttons:
            button.configure(state="normal" if idle else "disabled")
        for widget in self.editable:
            widget.configure(state="disabled" if self.busy else ("readonly" if isinstance(widget, ttk.Combobox) else "normal"))
        for widget in self.connection_entries:
            widget.configure(state="disabled" if self.busy or self.connected else "normal")
        self.connection_text.set(f"控制：{'已连接' if self.connected else '未连接'}　日志：{self.console_state}")

    def append_log(self, message):
        self.log_text.insert("end", message)
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 6000:
            self.log_text.delete("1.0", f"{lines - 5000}.0")
        self.log_text.see("end")

    def handle_event(self, kind, value):
        if kind == "log":
            self.append_log(value)
        elif kind == "connected":
            self.connected = value
            if value:
                self.notice.set("连接成功，选择采集长度和保存路径后开始")
                self.progress_text.set("已连接，等待采集或下载")
        elif kind == "console":
            self.console_state = value
        elif kind == "status":
            self.info = value
            state_names = {"empty": "可采集", "capturing": "采集中", "ready": "数据就绪（下载或释放）",
                           "failed": "采集失败", "unavailable": "等待启动检查"}
            self.board_text.set(f"板卡：{state_names.get(value['state'], value['state'])}　"
                                f"链路：{value.get('link_mbps', '—')} Mbps　"
                                f"采集编号：{value.get('capture_id') or '—'}")
            self.detail_text.configure(state="normal")
            self.detail_text.delete("1.0", "end")
            self.detail_text.insert("1.0", json.dumps(value, ensure_ascii=False, indent=2))
            self.detail_text.configure(state="disabled")
        elif kind == "phase":
            self.operation = "capture" if value == "capture" else ("download" if value == "download" else self.operation)
            self.progress.configure(value=0)
            self.stop_button.configure(text="终止采集" if value == "capture" else "暂停下载")
            self.progress_text.set({"connect": "正在连接…", "capture": "正在准备并采集…", "download": "正在校验文件并准备下载…"}[value])
        elif kind == "progress":
            done, total, rate, phase = value
            self.progress.configure(value=100 * done / total if total else 0)
            label = {"capture": "采集", "verify": "校验已有文件", "download": "下载", "complete": "保存完成"}[phase]
            self.progress_text.set(f"{label}：{done / 1048576:.3f} / {total / 1048576:.3f} MiB"
                                   + (f"　{rate / 1048576:.2f} MiB/s" if rate else ""))
            self.stop_button.configure(text="终止采集" if phase == "capture" else "暂停下载")
        elif kind == "saved":
            filename, metadata = value
            self.last_download = filename
            if self.spectrum_window is not None:
                self.spectrum_window.offer_download(filename)
            self.notice.set(f"已保存：{filename}（bin + json，校验通过）")
            self.append_log(f"\n保存完成：{filename}\nSHA256：{metadata['sha256']}\n")
        elif kind in ("error", "notice"):
            text = ("操作失败：" if kind == "error" else "") + value
            self.notice.set(text)
            self.append_log(f"\n{text}\n")
        elif kind == "done":
            self.busy = False
            self.operation = ""
            self.stop_button.configure(text="停止当前操作")
        self.update_buttons()

    def poll_events(self):
        if self.closing:
            return
        for _ in range(300):
            try:
                event = self.controller.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(*event)
        self.poll_after = self.root.after(50, self.poll_events)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.spectrum_window is not None:
            self.spectrum_window.close()
        if self.poll_after:
            self.root.after_cancel(self.poll_after)
            self.poll_after = None
        self.controller.close()
        self._wait_close(time.monotonic() + 1.5)

    def _wait_close(self, deadline):
        if not self.controller.thread.is_alive() or time.monotonic() >= deadline:
            self.destroy()
        else:
            self.close_after = self.root.after(50, lambda: self._wait_close(deadline))

    def destroy(self):
        for callback in (self.poll_after, self.close_after):
            if callback:
                try:
                    self.root.after_cancel(callback)
                except tk.TclError:
                    pass
        self.poll_after = self.close_after = None
        self.root.destroy()


def main():
    root = tk.Tk()
    Rx706App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
