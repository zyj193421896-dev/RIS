"""Actual Matplotlib event/transform checks for cursor zoom and free pan."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent,KeyEvent

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_plot_navigation import PlotNavigation


class PlotNavigationTests(unittest.TestCase):
    def setUp(self):
        self.figure=Figure(figsize=(8,5))
        self.canvas=FigureCanvasAgg(self.figure)
        self.ax=self.figure.add_subplot(111)
        self.ax.plot([0,10],[0,20]); self.ax.set(xlim=(0,10),ylim=(0,20))
        self.toolbar=SimpleNamespace(mode='',push_current=lambda:None)
        self.focus=[]
        self.nav=PlotNavigation(self.canvas,self.toolbar,self.focus.append)
        self.canvas.draw()

    def tearDown(self): self.nav.close()

    def mouse(self,name,x,y,**options):
        px,py=self.ax.transData.transform((x,y))
        return MouseEvent(name,self.canvas,px,py,**options)

    def test_wheel_keeps_cursor_anchor_and_home_restores_data(self):
        self.nav.scroll(self.mouse('scroll_event',2,5,button='up',step=1))
        np.testing.assert_allclose(self.ax.get_xlim(),[.4,8.4],atol=.03)
        np.testing.assert_allclose(self.ax.get_ylim(),[1,17],atol=.03)
        self.nav.key(KeyEvent('key_press_event',self.canvas,key='r'))
        np.testing.assert_allclose(self.ax.get_xlim(),[0,10])
        np.testing.assert_allclose(self.ax.get_ylim(),[0,20])

    def test_drag_translates_axes_and_stops_on_release(self):
        self.nav.press(self.mouse('button_press_event',2,5,button=1))
        self.nav.motion(self.mouse('motion_notify_event',3,7,button=1))
        np.testing.assert_allclose(self.ax.get_xlim(),[-1,9],atol=.05)
        np.testing.assert_allclose(self.ax.get_ylim(),[-2,18],atol=.05)
        self.nav.release()
        saved=self.ax.get_xlim()
        self.nav.motion(self.mouse('motion_notify_event',8,10,button=1))
        self.assertEqual(self.ax.get_xlim(),saved)

    def test_modifiers_and_toolbar_priority(self):
        self.nav.scroll(self.mouse('scroll_event',2,5,button='up',step=1,key='shift'))
        np.testing.assert_allclose(self.ax.get_ylim(),[0,20])
        self.nav.reset()
        self.nav.scroll(self.mouse('scroll_event',2,5,button='up',step=1,key='control'))
        np.testing.assert_allclose(self.ax.get_xlim(),[0,10])
        self.toolbar.mode='pan/zoom'
        saved=self.ax.get_ylim()
        self.nav.scroll(self.mouse('scroll_event',2,5,button='up',step=1))
        self.assertEqual(self.ax.get_ylim(),saved)

    def test_double_click_focus_callback_and_disconnect(self):
        self.nav.press(self.mouse('button_press_event',2,5,button=1,dblclick=True))
        self.assertEqual(self.focus,[self.ax])
        self.assertIsNone(self.nav.drag)
        self.nav.close()
        self.assertEqual(self.nav.connections,[])

    def test_constrained_layout_does_not_move_axes_during_drag(self):
        self.figure.set_layout_engine('constrained')
        self.ax.set_title('Stable plot during mouse dragging')
        self.canvas.draw()
        engine=self.figure.get_layout_engine()
        bounds=np.array(self.ax.get_position().bounds)
        start=self.mouse('button_press_event',2,5,button=1)
        self.nav.press(start)
        self.assertIsNone(self.figure.get_layout_engine())
        for offset in (15,30,50):
            event=MouseEvent('motion_notify_event',self.canvas,start.x+offset,start.y+offset,button=1)
            self.nav.motion(event)
            self.canvas.draw()
            np.testing.assert_allclose(self.ax.get_position().bounds,bounds,atol=1e-9)
        self.nav.release()
        self.assertIs(self.figure.get_layout_engine(),engine)
        self.assertIsNone(self.nav.drag)
        self.toolbar.mode='pan/zoom'
        self.nav.press(self.mouse('button_press_event',2,5,button=1,dblclick=True))
        self.assertFalse(self.focus)


if __name__=='__main__': unittest.main()
