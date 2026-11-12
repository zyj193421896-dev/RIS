"""RIS receiver with optional board capture and offline BIN workflows.

No automatic board connection, capture, or decode is performed on startup.
"""
import argparse
import tkinter as tk
from rx706_spectrum_gui import SpectrumWindow
from rx706_receiver_capture import ReceiverCaptureBridge


def create_app(filename=''):
    root = tk.Tk()
    root.withdraw()
    try:
        view = SpectrumWindow(root, filename=filename)
        view.board_capture = ReceiverCaptureBridge(view)
    except Exception:
        root.destroy()
        raise
    close_window = view.close
    closed = False
    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            view.board_capture.close()
        finally:
            try:
                close_window()
            finally:
                root.destroy()
    view.close = close
    view.window.protocol('WM_DELETE_WINDOW', close)
    view.window.title('RIS 收端 · PWM测速 / 256QAM验证 / 三用户验证')
    return root, view


def main(argv=None):
    parser = argparse.ArgumentParser(description='直接打开RIS收端三功能区')
    parser.add_argument('bin', nargs='?', default='', help='可选IQ bin；不自动推测发送配置')
    args = parser.parse_args(argv)
    root, _ = create_app(args.bin)
    root.mainloop()


if __name__ == '__main__':
    main()
