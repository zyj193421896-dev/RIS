"""Exercise real Tk controls/worker and render a local-file recovery preview.

All windows stay withdrawn. No board connection is made.
"""
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from matplotlib.backend_bases import MouseEvent

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow,RECOVERY_MODE,TONE_MODE

out=ROOT/'reports/gui_recovery_20261028'
out.mkdir(exist_ok=True)
root=tk.Tk(); root.withdraw()
errors=[]
root.report_callback_exception=lambda *args: errors.append(str(args))
view=SpectrumWindow(root,ROOT/'captures/5p802_switch100k_m50dbm.bin')
view.window.withdraw()

def until_idle():
    deadline=time.monotonic()+30
    while view.busy and time.monotonic()<deadline:
        root.update(); time.sleep(.01)
    root.update()
    assert not view.busy,'Analysis timed out'
    assert not errors,errors

try:
    view.constellation_mode.set(RECOVERY_MODE)
    view.on_mode_changed()
    until_idle()
    assert view.recovery is not None,view.status.get()
    r=view.recovery
    assert 170<abs(r.phase_difference_deg)<180
    view.figure.savefig(out/'recovered_gui_plot.png',dpi=150)
    result=dict(status=view.status.get(),summary=view.summary.get(),carrier_hz=r.carrier_hz,
                cycle_hz=r.cycle_hz,phase_difference_deg=r.phase_difference_deg,
                amplitude_difference_db=r.amplitude_difference_db,separation=r.separation,
                heldout_clock_agreement=r.heldout_clock_agreement,tk_version=root.tk.call('info','patchlevel'))
    navigation_out=ROOT/'reports/gui_navigation_20261028'
    navigation_out.mkdir(exist_ok=True)
    view.figure.savefig(navigation_out/'overview.png',dpi=120)
    overview_area=view.iq_axes.bbox.width*view.iq_axes.bbox.height
    view.double_click_plot(view.iq_axes)
    root.update(); view.canvas.draw()
    assert view.plot_view.get()=='星座'
    assert sum(ax.get_visible() for ax in view.figure.axes)==1
    single_area=view.iq_axes.bbox.width*view.iq_axes.bbox.height
    assert single_area>overview_area*2,(overview_area,single_area)
    xlim=view.iq_axes.get_xlim(); ylim=view.iq_axes.get_ylim()
    px,py=view.iq_axes.transData.transform(((xlim[0]+xlim[1])/2,(ylim[0]+ylim[1])/2))
    view.navigation.scroll(MouseEvent('scroll_event',view.canvas,px,py,button='up',step=1))
    root.update()
    assert abs(view.iq_axes.get_xlim()[1]-view.iq_axes.get_xlim()[0])<abs(xlim[1]-xlim[0])
    view.navigation.reset(); root.update()
    view.figure.savefig(navigation_out/'constellation_large.png',dpi=120)
    view.toggle_sidebar(); root.update()
    assert len(view.panes.panes())==1
    view.toggle_sidebar(); root.update()
    assert len(view.panes.panes())==2
    for name in ['频谱','功率','IQ 波形','相位']:
        view.set_plot_view(name); root.update()
        assert sum(ax.get_visible() for ax in view.figure.axes)==1
    view.double_click_plot(view.phase_axes); root.update()
    assert view.plot_view.get()=='总览'
    assert all(ax.get_visible() for ax in view.figure.axes)
    result['navigation_checks']=dict(single_plot_area_gain=single_area/overview_area,
        cursor_zoom=True,sidebar_toggle=True,all_single_views=True,double_click_restore=True)
    for phase_mode in ['展开相位','双通道相对相位','包裹相位 ±180°']:
        view.phase_mode.set(phase_mode); view.draw_constellation(); root.update()
    view.constellation_mode.set('原始 IQ'); view.on_mode_changed(); root.update()
    assert 'Raw block' in view.power_axes.get_title()
    view.constellation_mode.set(RECOVERY_MODE); view.on_mode_changed(); root.update()
    # A new fixed-state tone must not silently display cached recovered A/B states.
    view.offer_download(ROOT/'captures/5p802_all1_m50dbm.bin')
    until_idle()
    assert view.recovery is None,'Fixed tone was incorrectly forced into two states'
    assert '未锁定' in view.status.get(),view.status.get()
    result['fixed_tone_rejected']=True
    # The same fixed-state recording must succeed in the dedicated CW mode.
    view.constellation_mode.set(TONE_MODE); view.on_mode_changed(); until_idle()
    assert view.recovery is not None,view.status.get()
    assert view.recovery.kind=='tone'
    tone=view.recovery
    tone_out=ROOT/'reports/gui_tone_20261028'; tone_out.mkdir(exist_ok=True)
    view.figure.savefig(tone_out/'overview.png',dpi=120)
    view.set_plot_view('星座'); root.update()
    view.figure.savefig(tone_out/'constellation.png',dpi=120)
    result['tone_recovery']=dict(carrier_hz=tone.carrier_hz,
        magnitude_counts=float(abs(tone.centers[0])),phase_rms_deg=tone.phase_rms_deg,
        residual_frequency_hz=tone.residual_frequency_hz,output_rate_hz=tone.output_rate_hz,
        status=view.status.get(),summary=view.summary.get())
    (tone_out/'verification.json').write_text(json.dumps(result['tone_recovery'],ensure_ascii=False,indent=2),encoding='utf-8')
    view.set_plot_view('总览')
    # Recovery cache must not cross CW/periodic modes or different input files.
    view.offer_download(ROOT/'captures/5p802_switch100k_m50dbm.bin'); until_idle()
    assert view.recovery is None,'Periodic modulation was incorrectly accepted as fixed CW'
    view.constellation_mode.set(RECOVERY_MODE); view.on_mode_changed(); until_idle()
    assert view.recovery is not None and view.recovery.kind=='periodic',view.status.get()
    result['callback_errors']=errors
    assert not errors,errors
    (navigation_out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('status','summary','tone_recovery')},indent=2))
    print(json.dumps({k:v for k,v in result['tone_recovery'].items() if k not in ('status','summary')},indent=2))
finally:
    view.close(); root.destroy()
