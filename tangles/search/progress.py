from typing import Callable


PROGRESS_TYPE_SOMETHING_STARTING = 1
PROGRESS_TYPE_SOMETHING_FINISHED = 2

PROGRESS_TYPE_SWEEP_RUNNING = 3
PROGRESS_TYPE_UNCROSSING_RUNNING = 4
PROGRESS_TYPE_SEP_APPENDING_RUNNING = 5


TangleSearchProgressType = Callable[[str, ...], None]


class DefaultProgressCallback:
    """A default progress callable for the tangle search.
    This class can be used to stay informed about the progress of a tangle search.
    """

    def __init__(self, show_info_while_running: bool = True):
        self.num_starts = 0
        self.op_stack = []
        self.show_info_while_running = show_info_while_running

    def __call__(self, progress_type, **kwargs):
        pop = False
        if progress_type == PROGRESS_TYPE_SOMETHING_STARTING:
            self.op_stack.append(kwargs["info"])
        elif progress_type == PROGRESS_TYPE_SOMETHING_FINISHED:
            if "info" in kwargs:
                self.op_stack[-1] = kwargs["info"]
            pop = True
        elif progress_type == PROGRESS_TYPE_SWEEP_RUNNING:
            self.op_stack[-1] = (
                f"sweeping  (level {kwargs['level']+1}/{kwargs['sweep'].tree.number_of_separations})"
            )
        elif progress_type == PROGRESS_TYPE_UNCROSSING_RUNNING:
            self.op_stack[-1] = (
                f"uncrossing (corners added: {kwargs['num_corners_added']})"
            )
        elif progress_type == PROGRESS_TYPE_SEP_APPENDING_RUNNING:
            self.op_stack[-1] = (
                f"appending ({kwargs['num_seps_added']}/{kwargs['num_total_seps']})"
            )

        print("\r" + " " * 150 + "\r", end="")

        info = (
            f"[ tree height={sweep.tree.number_of_separations}, "
            + f"number of tangles={len(sweep.tree.maximal_tangles(sweep.tree.limit))}, "
            + f"limit={sweep.tree.limit} ]: "
            if (
                (self.show_info_while_running or pop) and (sweep := kwargs.get("sweep"))
            )
            else ""
        )
        print(f"\r{info}", end="")

        for s in self.op_stack:
            print(s, end=" -> " if s != self.op_stack[-1] else " ")
        print("                                   ", end="")
        if pop:
            self.op_stack.pop()
        if len(self.op_stack) == 0:
            print()


class SimpleProgressCallback:
    """A simpler (and much faster) progress callable for the tangle search."""

    def __init__(self, total_count: int):
        self.total_count = total_count
        self.counter = 0

    def __call__(self, progress_type, **kwargs):
        if progress_type == PROGRESS_TYPE_SOMETHING_STARTING:
            self.counter = 0
        elif progress_type != PROGRESS_TYPE_SOMETHING_FINISHED:
            self.counter += 1
            print(f"\r{self.counter}/{self.total_count}", end="")
        else:
            print()


# TODO: write a tqdm-version.
