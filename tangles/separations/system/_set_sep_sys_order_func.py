import numpy as np
from bisect import bisect_left
from ._set_system import SetSeparationSystemBase

class SetSeparationSystemOrderFunc:
    def __init__(self, sep_sys: SetSeparationSystemBase, order_func):
        self.sep_sys = sep_sys
        self.order_func = order_func
        self._orders = np.empty((2, 0))
        self._args = np.empty(0)
        self._args_inv = np.empty(0)
        self._update_order_cache()

    def _update_order_cache(self):
        beginning = self._orders.shape[1]
        num_seps = len(self.sep_sys)
        if beginning < num_seps:
            new_seps = self.sep_sys[beginning:]
            new_orders = np.empty((2, num_seps - beginning))
            new_orders[0, :] = -np.abs(new_seps.sum(axis=0))
            np.around(self.order_func(new_seps), 10, out=new_orders[1, :])  # this seems to fix the numerical problem. but is it always correct?
            self._orders = np.c_[self._orders, new_orders]                  # we do all this 'injectivity stuff' to be always correct, not 'nearly always' correct - the latter could be much easier..
            self._args = np.lexsort(self._orders)
            self._order_finetuning(beginning)
            self._args_inv = self._args.argsort()

    def _order_finetuning(self, first_new_idx): # we should rethink this, maybe there is a faster solution or a better implementation?
        sorted_orders = self._orders[:, self._args]
        sorted_orders_jumps = 1 + np.flatnonzero((sorted_orders[:, 1:] != sorted_orders[:, :-1]).any(axis=0))
        range_start = 0
        for jump in sorted_orders_jumps:
            if jump - range_start > 1 and np.any(self._args[range_start:jump] >= first_new_idx):
                seps = self.sep_sys[self._args[range_start:jump]]
                seps *= seps[0:1, :]
                self._args[range_start:jump] = self._args[range_start:jump][np.lexsort(seps)[::-1]]
            range_start = jump

    def get_sep(self, sep_ids):
        return self.sep_sys.seps[:, sep_ids]

    def get_order(self, sep_id):
        self._update_order_cache()
        return self._orders[1, sep_id]

    # it is very important, that sorted_sep_id_list really is sorted by order! TODO:
    def get_insertion_index(self, sorted_sep_id_list, sep_id_to_insert):
        self._update_order_cache()
        return bisect_left(self._args_inv[sorted_sep_id_list], self._args_inv[sep_id_to_insert])

    @property
    def sorted_ids(self):
        self._update_order_cache()
        return self._args

    def injective_order_value(self, sep_ids):
        self._update_order_cache()
        return self._args_inv[sep_ids]

    @property
    def sorted_orders(self):
        return self._orders[1, self.sorted_ids]

    @property
    def unsorted_orders(self):
        return self._orders[1, :]

    def __len__(self):
        return len(self.sep_sys)