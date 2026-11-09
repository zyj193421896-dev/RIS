"""Capture-only GUI. No NumPy, Matplotlib, demodulators or IDE are needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'app'))
import tkinter as tk
from rx706_gui import Rx706App


class CaptureApp(Rx706App):
    def _build(self):
        super()._build()
        self.spectrum_button.destroy()


def main():
    root = tk.Tk()
    app = CaptureApp(root)
    root.title('ZC706 IQ 采集与下载 · 独立版')
    app.notice.set('有线网卡设为192.168.1.20/24；连接并下载后，将.bin与同名.bin.json一起拷回分析电脑。')
    root.mainloop()


if __name__ == '__main__':
    main()
