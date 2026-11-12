"""Exercise plotting and result delivery with Agg, without a desktop/Tcl runtime."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import queue
import threading
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import SpectrumWindow


class Value:
    def __init__(self, value=None): self.value=value
    def get(self): return self.value
    def set(self, value): self.value=value


def headless_window():
    view=SpectrumWindow.__new__(SpectrumWindow)
    view.closed=False; view.busy=True; view.result=None; view.recovery=None
    view.events=queue.Queue(); view.cancel=threading.Event()
    view._pwm_dialog=view._pwm_dialog_result=None
    view.controls=[]; view.window=Mock(); view.toolbar=Mock()
    view.analyze_button=Mock(); view.stop_button=Mock()
    view.rate=Value(); view.status=Value(); view.summary=Value()
    view.filename=Value(); view.auto=Value(True)
    view.constellation_mode=Value('原始 IQ'); view.pending_download=None
    for name, value in dict(pwm2_training_enabled=False, pwm_reference_enabled=False, pwm_frame_enabled=False,
                           pwm2_clock_width=True, pwm2_decision_mode='连续脉宽',
                           pwm2_mapping='高片数减前高', pwm2_input_mode='时间片协议').items():
        setattr(view, name, Value(value))
    view.cycle_khz=Value('100'); view.recovery_ms=Value('1000'); view.center_hint=Value('自动')
    view.plot_view=Value('总览'); view._active_plot_name='总览'; view._overview_limits=None
    view.figure=Figure(figsize=(12,8),layout='constrained')
    view._create_plot_axes()
    view.canvas=FigureCanvasAgg(view.figure)
    view.refresh_parameter_visibility = Mock()
    return view


class SignalViewTests(unittest.TestCase):
    def test_worker_plot_modes_and_result_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tone.bin'; n=4096; t=np.arange(n*2)
            x=512*np.exp(2j*np.pi*125*t/n)
            np.rint(np.column_stack([x.real,x.imag,x.real,x.imag])).astype('<i2').tofile(p)
            view=headless_window()
            view._analyze_worker(p,dict(nfft=n,averages=2))
            view.poll()
            self.assertIsNotNone(view.result)
            self.assertEqual(len(view.figure.axes),3)
            self.assertIn('总功率',view.status.get())
            self.assertIn('非 dBm',view.summary.get())
            view.constellation_mode.set('原始 IQ')
            view.draw_constellation()
            points=view.iq_axes.collections[0].get_offsets()
            self.assertGreater(np.std(points[:,1]),100)
            self.assertFalse(hasattr(view,'power_axes'))
            self.assertFalse(hasattr(view,'phase_axes'))
            for name in ('星座','IQ 波形','频谱','总览'):
                view.set_plot_view(name)
                self.assertEqual(sum(ax.get_visible() for ax in view.figure.axes),3 if name == '总览' else 1)
            view.figure.savefig(Path(tmp)/'plot.png')

    def test_stop_retires_a_result_already_queued(self):
        view=headless_window()
        view.events.put(('result',(object(),None,None,None)))
        view.cancel.set(); view.poll()
        self.assertIsNone(view.result); self.assertIsNone(view.recovery)
        self.assertFalse(view.busy); self.assertIn('停止',view.status.get())

    def test_busy_download_is_delivered_after_result(self):
        view=headless_window()
        view.offer_download('next.bin')
        self.assertEqual(view.pending_download,'next.bin')
        view.run_workflow=Mock()
        view.events.put(('cancelled','stopped'))
        view.poll()
        self.assertEqual(view.filename.get(),'next.bin')
        view.run_workflow.assert_called_once()


if __name__=='__main__': unittest.main()
