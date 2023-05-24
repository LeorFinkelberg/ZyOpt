from zyopt._base_model import Model
from zyopt.common.logger import make_logger

logger = make_logger(__file__)


class FixBinaryIntegerVarsInRelaxSolModel(Model):
    """
    Fix Binary and Integer Variables in Relax Solution with Perturbation
    """

    def __init__(
        self,
        mode: str,
        path_to_problem: str,
        path_to_relax_params: str,
        path_to_milp_params: str,
        binary: bool = True,
        integer: bool = True,
        fix_vars_lower_threshold: float = 0.0,
        fix_vars_upper_threshold: float = 0.0,
        rounds: int = 100,
        distr_lower_threshold: float = -0.9,
        dist_upper_threshold: float = 0.1,
    ):
        pass
