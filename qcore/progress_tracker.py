import time

import numpy as np


def pp_time(seconds_remaining, space=5):
    """
    Pretty prints the given time in the largest useful unit
    If you get to years, good luck
    :param seconds_remaining: The number of seconds to be pretty printed
    :param space: The number of characters the number should take. Allows for constant width printing, potentially more
    visually appealing
    """
    time_counters = [
        ("s", 60),
        ("m", 60),
        ("h", 24),
        ("d", 30),
        (" months", 12),
    ]
    time_remaining = seconds_remaining
    for s, t in time_counters:
        if time_remaining > t:
            time_remaining /= t
        else:
            return f"{time_remaining: {space}.2f}{s}"
    return f"{time_remaining: {space}.2f} years"


class ProgressTracker:
    """
    A progress tracker that prints the current status of a job being run.
    Requires knowing how many tasks/items are going to be run and computes estimated time remaining assuming all items
    are the same size.
    Can be used as a context manager where the update function is assigned to the named variable.
    Note that using the context manager prevents reviewing the times after running
    Example:
    n_items = len(work)
    with ProgressTracker(n_items) as pt:
        for i, job in enumerate(work):
            func(job)
            pt(i)
    """

    def __init__(self, n_items, percent_increment=10, print_func=print):
        """
        Creates the progress tracker
        :param n_items: the number of items to be worked on.
        :param percent_increment: How often to print the progress. Will output when the percent complete first hits each
        integer multiple of this value
        :param print_func: The function to call when outputting the progress. Allows for logging functions to be used
        instead of print
        """
        self.n_items = n_items
        self.percent_increment = percent_increment
        self.current_increment = percent_increment
        self.started = False
        self.times = []
        self.print_func = print_func

    def __enter__(self):
        """
        Context manager entry function
        Starts the timer and returns the update function
        """
        self._start()
        return self.update

    def _start(self):
        """
        Start the timer
        Silently exits if called again
        """
        if not self.started:
            self.times.append(time.time())
            self.started = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method
        Converts the times from a list to an array
        """
        self.update(self.n_items)
        self._stop()

    def _stop(self):
        """
        Converts the list of absolute times to an array of times relative to the start time
        Allows for run time analysis later if required
        Note that times is not available if the tracker is used with the context manager
        """
        self.times = np.asarray(self.times) - self.times[0]

    def update(self, update_index):
        """
        Updates the tracker with the index of the item currently being run
        Adds the time to the list of times and outputs a string with percent completion, time taken since last update,
        time take since start, estimated time remaining and estimated total time
        """
        if not self.started:
            self._start()

        percent_done = 100 * update_index / self.n_items
        if percent_done >= self.current_increment:
            self.times.append(time.time())
            while percent_done >= self.current_increment:
                self.current_increment += self.percent_increment

            elapsed_time = self.times[-1] - self.times[0]
            step_time = self.times[-1] - self.times[-2]
            remaining_time = elapsed_time * self.n_items / update_index - elapsed_time
            self.print_func(
                f"{percent_done:5.1f}% complete. Time taken for this block: {pp_time(step_time)}. "
                f"Time elapsed: {pp_time(elapsed_time)}. Time remaining: {pp_time(remaining_time)}. "
                f"Total time: {pp_time(elapsed_time + remaining_time)}."
            )
