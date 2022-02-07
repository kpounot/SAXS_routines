"""Class for interactive range selection."""


import numpy as np


class RangeSelector:
    def __init__(self, fig, sample):
        self.fig = fig
        self.sample = sample
        self.new_sample = None
        self.new_buffer = None

        self.ax = self.fig.axes[0]
        self.ybound = None
        self.ax.set_autoscale_on(False)

        self.sample_ax = list(sample.shape).index(
            self.ax.lines[-1].get_xdata().size
        )

        self.line1 = None
        self.line2 = None
        self.fill = None
        self.button_pressed = False

        self.buffer_line1 = None
        self.buffer_line2 = None
        self.buffer_fill = None
        self.ctrl_pressed = False

        self.range = None

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.update()

    def on_click(self, event):
        if event.button == 3:
            self.clear_selection()
            self.button_pressed = True
            self.ybound = self.ax.get_ybound()
            if self.ctrl_pressed:
                self.buffer_line1 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:orange"
                )
                self.buffer_line2 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:orange"
                )
                self.buffer_fill = self.ax.fill_betweenx(
                    self.ybound,
                    self.buffer_line1.get_xdata(),
                    self.buffer_line2.get_xdata(),
                    color="tab:orange",
                    alpha=0.2,
                )
            else:
                self.line1 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:blue"
                )
                self.line2 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:blue"
                )
                self.fill = self.ax.fill_betweenx(
                    self.ybound,
                    self.line1.get_xdata(),
                    self.line2.get_xdata(),
                    color="tab:blue",
                    alpha=0.2,
                )
        else:
            return

        self.update()

    def on_release(self, event):
        if self.button_pressed:
            if self.ctrl_pressed:
                left = self.buffer_line1.get_xdata()[0]
                right = self.buffer_line2.get_xdata()[0]
                self.range = np.sort([left, right])
                if self.sample.shape[self.sample_ax] == self.sample.q.size:
                    self.new_buffer = self.sample.get_q_range(*self.range)
                else:
                    self.new_buffer = self.sample.get_time_range(*self.range)

            else:
                left = self.line1.get_xdata()[0]
                right = self.line2.get_xdata()[0]
                self.range = np.sort([left, right])
                if self.sample.shape[self.sample_ax] == self.sample.q.size:
                    self.new_sample = self.sample.get_q_range(*self.range)
                else:
                    self.new_sample = self.sample.get_time_range(*self.range)

            self.button_pressed = False

    def on_motion(self, event):
        if self.button_pressed:
            if self.ctrl_pressed:
                self.buffer_line2.remove()
                self.buffer_line2 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:orange"
                )
                self.buffer_fill.remove()
                self.buffer_fill = self.ax.fill_betweenx(
                    self.ybound,
                    self.buffer_line1.get_xdata(),
                    self.buffer_line2.get_xdata(),
                    color="tab:orange",
                    alpha=0.2,
                )
            else:
                self.line2.remove()
                self.line2 = self.ax.axvline(
                    event.xdata, ls=":", color="tab:blue"
                )
                self.fill.remove()
                self.fill = self.ax.fill_betweenx(
                    self.ybound,
                    self.line1.get_xdata(),
                    self.line2.get_xdata(),
                    color="tab:blue",
                    alpha=0.2,
                )

            self.update()

    def on_key_press(self, event):
        if event.key == "control":
            self.ctrl_pressed = True
        else:
            return

    def on_key_release(self, event):
        if event.key == "control":
            self.ctrl_pressed = False
        else:
            return

    def clear_selection(self):
        if self.ctrl_pressed:
            if self.buffer_line1 is not None:
                self.buffer_line1.remove()
                self.buffer_line1 = None
            if self.buffer_line2 is not None:
                self.buffer_line2.remove()
                self.buffer_line2 = None
            if self.buffer_fill is not None:
                self.buffer_fill.remove()
                self.buffer_fill = None
        else:
            if self.line1 is not None:
                self.line1.remove()
                self.line1 = None
            if self.line2 is not None:
                self.line2.remove()
                self.line2 = None
            if self.fill is not None:
                self.fill.remove()
                self.fill = None

    def update(self):
        self.fig.canvas.draw()
