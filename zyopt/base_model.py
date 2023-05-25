import abc

import pandas as pd


class Model(abc.ABC):
    @abc.abstractmethod
    def optimize(self):
        """
        Optimize the problem
        """

    @staticmethod
    def _make_mask(base_for_fix: pd.Series, lower_threshold: float, upper_threshold: float) -> pd.Series:
        """
        Makes bool mask
        """
        if lower_threshold == upper_threshold:
            # lower_threshold = upper_threshold = 0 or lower_threshold = upper_threshold = 1
            fix_value = lower_threshold
            mask = base_for_fix == fix_value
        elif upper_threshold is None:
            # Example: lower_threshold = 0.05 and upper_threshold = NULL
            base_for_fix[base_for_fix <= lower_threshold] = 0.0
            mask = base_for_fix == 0.0
        elif lower_threshold is None:
            # Example: upper_threshold = 0.95 and lower_threshold = NULL
            base_for_fix[base_for_fix >= upper_threshold] = 1.0
            mask = base_for_fix == 1.0
        else:
            # Example: lower_threshold = 0.05 and upper_threshold = 0.95
            base_for_fix[base_for_fix <= lower_threshold] = 0.0
            mask_for_zero = base_for_fix == 0.0

            base_for_fix[base_for_fix >= upper_threshold] = 1.0
            mask_for_one = base_for_fix == 1.0

            mask = mask_for_zero | mask_for_one

        return mask
