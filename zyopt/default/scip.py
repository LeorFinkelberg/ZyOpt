import typing as t
from pathlib2 import Path
from zyopt.common.constants import *
from zyopt.common.logger import make_logger
import pyscipopt

logger = make_logger(__file__)


class Scip:
    def __init__(
        self,
        *,
        mode: str,
        path_to_problem: str,
        path_to_params: str,
    ):
        self.mode = mode
        self.path_to_problem = Path(path_to_problem)
        self.path_to_params = Path(path_to_params)

        self._model = pyscipopt.Model()
        self._model.readProblem(self.path_to_problem)
        self._model.readParams(self.path_to_params)

    def optimize(self):
        """
        Optimize the problem
        """
        if self.mode == SOLVER_MODE_RELAX:
            _all_vars: t.List[pyscipopt.scip.Variable] = self._model.getVars()
            _bin_int_vars: t.List[pyscipopt.scip.Variable] = [
                var for var in _all_vars if var.vtype() != VAR_TYPE_CONTINUOUS
            ]

            var: pyscipopt.scip.Variable
            for var in _bin_int_vars:
                self._model.chgVarType(var, VAR_TYPE_CONTINUOUS)

        logger.info(f"Running SCIP in {self.mode.upper()} mode ...")
        self._model.optimize()

    def get_status(self) -> str:
        """
        Retrieve solution status
        """
        return self._model.getStatus()
