"""Explicit transmitter-frame editor. No waveform inference or receiver provenance."""
import copy
import json
from pathlib import Path

from rx706_pwm_frames import parse_symbols
from rx706_pwm_pulse import validate_pulse_options


def sequence_text(value):
    return value if isinstance(value, str) else ' '.join(map(str, value))


def initial_fields(profile=None):
    """Copy sender fields only; old references/calibration require fresh opt-in."""
    p = copy.deepcopy(profile or {})
    users = p.get('users', [])
    if int(p.get('user_count', len(users) or 1)) != 1 or len(users) > 1:
        raise ValueError('快速向导只支持单用户，未修改任何配置。多用户请使用原“用户身份与净荷 / 已知数据”和保存功能。')
    user = users[0] if users else {}
    cal = p.get('cycle_calibration') or {}
    table = p.get('codebook', '')
    if not table and users:
        compiled = validate_pulse_options(dict(p, step_percent=p['duty_step'],
            symbol_count=p['duty_count'], frame_options=dict(p,
            reference_format='bits' if p.get('reference_format') in ('bits', '二进制比特') else 'symbols')),
            float(p['symbol_khz'])*1000)
        table = compiled['codebook']
    if not isinstance(table, str):
        table = '\n'.join(f'{d:g} {w:g}' for d, w in table)
    return dict(name=user.get('name', 'U1'), identity=sequence_text(user.get('identity', '')),
        payload_symbols=str(user.get('payload_symbols', '')), symbol_khz=str(p.get('symbol_khz', '170')),
        identity_errors=str(p.get('identity_errors', 1)),
        duty_step=str(p.get('duty_step', '5')), min_high_chips=str(p.get('min_high_chips', '5')),
        low_guard_chips=str(p.get('low_guard_chips', '5')), joint_delay=bool(p.get('joint_delay', True)),
        codebook=table, reference=sequence_text(user.get('reference', '')),
        reference_enabled=False, reference_format='bits' if p.get('reference_format') in ('bits', '二进制比特') else 'symbols',
        reference_repeat=bool(p.get('reference_repeat', True)), reference_reset=bool(p.get('reference_reset', True)),
        calibration_enabled=False, calibration_positions=' '.join(str(int(x)+1) for x in cal.get('payload_indices', [])),
        calibration_symbols=sequence_text(cal.get('symbols', '')), average_blocks=str(cal.get('average_blocks', 64)))


def build_profile(fields):
    """Validate explicit inputs and return a portable, receiver-independent JSON."""
    f = dict(fields)
    table = f['codebook'].strip()
    if not table:
        raise ValueError('请填写真实发端码表；向导不会自动猜测码表。')
    if f.get('reference_enabled') and str(f.get('reference', '')).strip().startswith('@'):
        raise ValueError('请粘贴已知参考内容，不使用外部文件路径，以便JSON独立复用。')
    cal = {}
    if f.get('calibration_enabled'):
        positions = parse_symbols(f['calibration_positions'], 1000001)
        if any(x < 1 for x in positions):
            raise ValueError('校准位置从1开始；例如净荷第4个符号填写4。')
        cal = dict(user_id=1, payload_indices=[x-1 for x in positions],
            symbols=list(parse_symbols(f['calibration_symbols'], 32768)), average_blocks=f['average_blocks'])
    p = dict(schema_version=3, mode='脉冲宽度与时延', user_count=1,
        users=[dict(name=f.get('name', 'U1') or 'U1', identity=f['identity'],
            payload_symbols=f['payload_symbols'], reference=f['reference'] if f.get('reference_enabled') else '')],
        symbol_khz=float(f['symbol_khz']), duty_step=float(f['duty_step']), duty_count=2,
        min_high_chips=f['min_high_chips'], low_guard_chips=f['low_guard_chips'],
        joint_delay=bool(f['joint_delay']), codebook=table, cycle_calibration=cal,
        reference_enabled=bool(f.get('reference_enabled')), reference_format=f['reference_format'],
        reference_repeat=bool(f['reference_repeat']), reference_reset=bool(f['reference_reset']),
        search_symbols=4096, identity_errors=f['identity_errors'], clock_width=True)
    # Count explicit rows for width-only codebooks as well as joint codebooks.
    p['duty_count'] = len([r for r in table.replace(';', '\n').splitlines() if r.strip()])
    compiled = validate_pulse_options(dict(p, step_percent=p['duty_step'],
        symbol_count=p['duty_count'], frame_options=p), p['symbol_khz']*1000)
    user = compiled['frame_options']['users'][0]
    if p['reference_enabled'] and not user['reference']:
        raise ValueError('已启用BER参考但内容为空；填写真实参考或取消勾选。')
    # Materialize normalized values, never a reference path or capture metadata.
    p.update(min_high_chips=compiled['min_high_chips'], low_guard_chips=compiled['low_guard_chips'],
        duty_count=compiled['symbol_count'], codebook=[list(row) for row in compiled['codebook']],
        cycle_calibration=compiled['cycle_calibration'])
    p['identity_errors'] = compiled['frame_options']['identity_errors']
    p['users'] = [dict(name=user['name'], identity=list(user['identity']), payload_symbols=user['payload_symbols'],
        reference=(list(user['reference']) if p['reference_format'] == 'symbols' else user['reference']) if p['reference_enabled'] else '')]
    return p


def preview_profile(profile):
    p = profile
    user = p['users'][0]
    table = p['codebook']
    total = round(100/p['duty_step'])
    period_us = 1000/p['symbol_khz']
    chip_us = period_us/total
    cal = p['cycle_calibration']
    reserved = dict(zip(cal.get('payload_indices', []), cal.get('symbols', [])))
    identity = user['identity']
    payload = user['payload_symbols']
    lines = ['发端帧配置预览（不是实测信号认证）',
        f"设定码元率：{p['symbol_khz']:g} kBaud；周期：{period_us:.9g} µs；{total}片/周期；每片{chip_us:.9g} µs。",
        f"一帧：身份前导 {len(identity)} + 净荷区 {payload} = {len(identity)+payload} 个PWM周期。",
        f"净荷区含保留校准 {len(reserved)} 个；参与数据评分的位置 {payload-len(reserved)} 个。",
        '无公共训练/缓冲；按“身份前导 → 净荷区”循环。符号编号从0开始，界面位置从1开始。',
        '前导由你指定，不自动搜索或生成。任何前导均可能出现在任意净荷中，不能保证无误锁。',
        '码元率是发端设定值；当前bin的实测码元率/BER须分析后得到。',
        ('已启用真实参考比对；参考未经独立认证，擦除/未覆盖另计。' if p['reference_enabled'] else 'BER未测：未启用已知净荷参考，不能把解码成功当作BER=0。'),
        f"参考格式={p['reference_format']}，循环={p['reference_repeat']}，每块重置={p['reference_reset']}。",
        '参考按整个净荷区定位（包含保留校准位置）；校准位置不参与数据BER。',
        '未携带采样文件哈希、A/B极性或固定接收设置；应用后须重新分析。',
        f"身份容错：{p['identity_errors']}个；初始搜索4096周期；接收容差默认0.4片，可用自动接收调试重新搜索。", '', '身份前导：']
    def describe(symbol):
        d, w = table[symbol]
        return f'符号{symbol} → 延迟{d:g}片/{d*chip_us:.9g}µs，高{w:g}片/{w*chip_us:.9g}µs，余低{total-d-w:g}片'
    lines += [f'  前导第{i+1}个：{describe(s)}' for i, s in enumerate(identity)]
    lines += ['', '净荷区布局（位置从1开始）：']
    start = 0
    for pos, symbol in sorted(reserved.items()):
        if pos > start:
            lines.append(f'  第{start+1}～{pos}个：数据评分区')
        lines.append(f'  第{pos+1}个：保留校准，{describe(symbol)}')
        start = pos+1
    if start < payload:
        lines.append(f'  第{start+1}～{payload}个：数据评分区')
    if cal:
        lines.append(f"校准平均块数：{cal['average_blocks']}。这些位置必须确实发送指定符号。")
    lines += ['', '完整码表（每行对应符号编号）：']
    lines += ['  '+describe(i) for i in range(len(table))]
    return '\n'.join(lines)


def save_new_profile(filename, profile):
    """Exclusive creation: never overwrite a user file, even after dialog approval."""
    path = Path(filename)
    if path.suffix.lower() != '.json':
        raise ValueError('文件名须以.json结尾。')
    # Validate again before any write, including callers outside the dialog.
    fields = initial_fields(profile)
    fields.update(reference_enabled=profile['reference_enabled'],
                  calibration_enabled=bool(profile.get('cycle_calibration')))
    checked = build_profile(fields)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(checked, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path


class FrameBuilderDialog:
    """Draft-only until explicit save/apply. Preview is always available in-dialog."""
    def __init__(self, parent, profile=None, apply_callback=None, fields=None):
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        self.tk = tk
        self.apply_callback = apply_callback
        self.seed = copy.deepcopy(fields) if fields is not None else initial_fields(profile)
        self.window = tk.Toplevel(parent)
        self.window.title('生成发端帧JSON · 明确前导与净荷')
        self.window.geometry('920x740')
        self.window.transient(parent)
        box = ttk.Frame(self.window, padding=12)
        box.pack(fill='both', expand=True)
        ttk.Label(box, text='草稿来自当前单用户配置；旧参考/循环校准默认关闭，须核对真实发端后主动启用。\n取消不会改变主窗口。清空新建不会自动生成前导或参考。', wraplength=870).pack(fill='x')
        top = ttk.Frame(box)
        top.pack(fill='x', pady=6)
        ttk.Button(top, text='清空新建', command=self.reset_blank).pack(side='left')
        ttk.Button(top, text='恢复打开时的草稿', command=lambda: self.populate(self.seed)).pack(side='left')
        self.notebook = ttk.Notebook(box)
        self.notebook.pack(fill='both', expand=True)
        pages = []
        for title in ('1 前导与时序', '2 实际码表', '3 参考与校准（可选）', '4 帧预览'):
            page = ttk.Frame(self.notebook, padding=10)
            self.notebook.add(page, text=title)
            pages.append(page)
        self.vars, self.texts = {}, {}
        def entry(page, key, label):
            ttk.Label(page, text=label).pack(anchor='w', pady=(6, 1))
            self.vars[key] = tk.StringVar(self.window)
            ttk.Entry(page, textvariable=self.vars[key]).pack(fill='x')
        def check(page, key, label):
            self.vars[key] = tk.BooleanVar(self.window)
            ttk.Checkbutton(page, variable=self.vars[key], text=label).pack(anchor='w', pady=3)
        for key, label in (('name', '用户名称'), ('identity', '你指定的前导（至少4个符号编号，用空格分隔）'),
                ('payload_symbols', '每帧净荷区长度（包含保留校准位置，单位：符号）'),
                ('identity_errors', '接收识别允许的前导错误符号数（小于前导长度一半）'),
                ('symbol_khz', '发端设定码元率（kBaud，不是bit/s）'), ('duty_step', '每片占周期百分比（如5表示20片）'),
                ('min_high_chips', '最小高电平片数'), ('low_guard_chips', '末尾预留低电平片数')):
            entry(pages[0], key, label)
        check(pages[1], 'joint_delay', '联合时延/脉宽码表（相同宽度、不同时延可表示不同符号）')
        ttk.Label(pages[1], text='每行：延迟片数 高电平片数；第1行=符号0，第2行=符号1……\n必须与发端一致，不自动补全或猜测。').pack(anchor='w')
        ttk.Button(pages[1], text='导入MT文本（高电平%,时延%）…', command=self.import_mt).pack(anchor='w', pady=4)
        self.texts['codebook'] = scrolledtext.ScrolledText(pages[1], height=18)
        self.texts['codebook'].pack(fill='both', expand=True)
        check(pages[2], 'reference_enabled', '我已确认以下参考是真实发送数据，启用BER测试（不选=BER未测）')
        self.vars['reference_format'] = tk.StringVar(self.window)
        ttk.Combobox(pages[2], textvariable=self.vars['reference_format'], values=('symbols', 'bits'), state='readonly').pack(fill='x')
        ttk.Label(pages[2], text='symbols=符号编号；bits=标签比特。粘贴整个净荷区参考，包含校准位置；不可把未知数据当参考。').pack(anchor='w')
        self.texts['reference'] = scrolledtext.ScrolledText(pages[2], height=5)
        self.texts['reference'].pack(fill='x')
        check(pages[2], 'reference_repeat', '循环重复参考（短参考将重复，不是自动补齐真实数据）')
        check(pages[2], 'reference_reset', '每块从参考起点开始（否则按用户连续）')
        check(pages[2], 'calibration_enabled', '我确认发端固定发送以下4个已知校准符号（不属于数据评分）')
        for key, label in (('calibration_positions', '4个净荷位置（从1开始，不含前导；例如4 5 6 7）'),
                           ('calibration_symbols', '对应4个真实符号编号（从0开始，须覆盖至少两种脉宽）'),
                           ('average_blocks', '校准平均块数（1～4096）')):
            entry(pages[2], key, label)
        self.preview = scrolledtext.ScrolledText(pages[3], wrap='word')
        self.preview.pack(fill='both', expand=True)
        self.note = tk.StringVar(self.window, '请核对实际发端组合后预览。')
        ttk.Label(box, textvariable=self.note, wraplength=870).pack(fill='x', pady=6)
        buttons = ttk.Frame(box)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='验证 / 刷新预览', command=self.show_preview).pack(side='left')
        ttk.Button(buttons, text='另存新JSON…', command=self.save).pack(side='left')
        ttk.Button(buttons, text='明确应用到当前分析', command=self.apply).pack(side='left')
        ttk.Button(buttons, text='关闭 / 取消', command=self.window.destroy).pack(side='right')
        self.populate(self.seed)
        for key in ('identity', 'payload_symbols', 'symbol_khz', 'duty_step', 'min_high_chips', 'low_guard_chips', 'joint_delay'):
            self.vars[key].trace_add('write', self.frame_changed)
        self.texts['codebook'].bind('<<Modified>>', self.codebook_changed)

    def populate(self, fields):
        self._populating = True
        for key, var in self.vars.items():
            var.set(fields[key])
        for key, text in self.texts.items():
            text.delete('1.0', 'end')
            text.insert('1.0', fields[key])
            text.edit_modified(False)
        self._populating = False
        self.preview.delete('1.0', 'end')
        self.note.set('草稿尚未应用。旧参考与校准均未启用；如需使用请核对后勾选。')

    def import_mt(self, filename=None):
        from tkinter import filedialog, messagebox
        from rx706_pwm_symbols import import_mt_fields
        filename = filename or filedialog.askopenfilename(parent=self.window,
            title='导入(高电平百分比,时延百分比);文本，不是片数码表', filetypes=[('MT文本', '*.txt')])
        if not filename:
            return False
        try:
            fields = import_mt_fields(self.fields(), Path(filename).read_text(encoding='utf-8-sig'))
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            messagebox.showerror('码表未导入', str(exc), parent=self.window)
            return False
        self.populate(fields)
        self.note.set('已按文件顺序导入，并按物理波形重映射前导；旧参考和校准已清除。净荷长度已清空，请填写实际发端长度。')
        return True

    def reset_blank(self):
        self.populate(initial_fields())

    def frame_changed(self, *args):
        if self._populating:
            return
        self.vars['reference_enabled'].set(False)
        self.vars['calibration_enabled'].set(False)
        self.preview.delete('1.0', 'end')
        self.note.set('帧组合已改变：参考和校准已关闭，请重新核对后勾选并刷新预览。')

    def codebook_changed(self, event=None):
        if self.texts['codebook'].edit_modified():
            self.texts['codebook'].edit_modified(False)
            self.frame_changed()

    def fields(self):
        values = {key: var.get() for key, var in self.vars.items()}
        values.update({key: text.get('1.0', 'end').strip() for key, text in self.texts.items()})
        return values

    def validated(self):
        from tkinter import messagebox
        try:
            return build_profile(self.fields())
        except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
            messagebox.showerror('帧配置错误', str(exc), parent=self.window)
            return None

    def show_preview(self):
        profile = self.validated()
        if profile is not None:
            self.preview.delete('1.0', 'end')
            self.preview.insert('1.0', preview_profile(profile))
            self.notebook.select(3)
            self.note.set('语法/物理边界验证通过，不代表实际发端或当前bin已经认证。')
        return profile

    def save(self):
        from tkinter import filedialog, messagebox
        profile = self.show_preview()
        if profile is None:
            return
        filename = filedialog.asksaveasfilename(parent=self.window, title='另存新的发端帧JSON（禁止覆盖）',
            initialfile='my_pwm_frame.json', defaultextension='.json', filetypes=[('PWM帧JSON', '*.json')])
        if not filename:
            return
        try:
            path = save_new_profile(filename, profile)
        except (OSError, ValueError) as exc:
            messagebox.showerror('未保存', str(exc)+'\n请选择尚不存在的新文件名。', parent=self.window)
            return
        self.note.set(f'已另存 {path}；主窗口配置未改变。可明确应用，或以后载入此JSON。')

    def apply(self):
        profile = self.show_preview()
        if profile is not None and self.apply_callback is not None:
            if self.apply_callback(copy.deepcopy(profile)):
                self.note.set('已明确应用到主窗口，旧分析已失效。请重新分析当前bin；如需复用请另存JSON。')
