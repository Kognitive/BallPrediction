import matplotlib.pyplot as plt
import matplotlib


class ThreadedPause:
    @staticmethod
    def pause(interval: float) -> None:
        """
        Pauses for the given interval without freezing figures.
        :param interval: The pause duration.
        """
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            fig_manager = matplotlib._pylab_helpers.Gcf.get_active()
            if fig_manager is not None:
                canvas = fig_manager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return
