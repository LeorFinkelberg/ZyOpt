import typing as t

import pyscipopt
from pathlib2 import Path

from zyopt._base_model import Model
from zyopt.common.constants import *
from zyopt.common.logger import make_logger
from zyopt.config import PYSCIPOPT_APACHE_2_0_LICENSE_VERSION

logger = make_logger(__file__)


class ScipModel(Model):
    """
    Simple wrapper for SCIP solver
    """

    def __init__(
        self,
        *,
        mode: str,
        path_to_problem: str,
        path_to_params: str,
    ):
        PYSCIPOPT_CURRENT_VERSION = tuple(map(int, pyscipopt.__version__.split(".")))
        if PYSCIPOPT_CURRENT_VERSION < PYSCIPOPT_APACHE_2_0_LICENSE_VERSION:
            logger.warning(
                f"You are using SCIP version {'.'.join(map(str, PYSCIPOPT_CURRENT_VERSION))}, "
                "which is only available under ZIB ACADEMIC LICENSE. "
                "See https://www.scipopt.org/academic.txt"
            )
        self.mode = mode
        self.path_to_problem = Path(path_to_problem)
        self.path_to_params = Path(path_to_params)

        self._model = pyscipopt.Model()
        logger.info(f"Reading problem: {self.path_to_problem}")
        self._model.readProblem(self.path_to_problem)
        logger.info(f"Reading params: {self.path_to_params}")
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

    def get_best_sol(self) -> dict:
        """
        Get the best SCIP sol in dict format
        """

        sol: pyscipopt.scip.Solution = self._model.getBestSol()
        vars_: t.Iterable[pyscipopt.scip.Variable] = self._model.getVars()

        return {var.name: self._model.getSolVal(sol, var) for var in vars_}

    def get_status(self) -> str:
        """
        Retrieve solution status
        """
        return self._model.getStatus()

    def get_params(self) -> dict:
        """
        Gets the values of all parameters as a dict mapping parameter names
        to their values
        """
        return self._model.getParams()

    def write_best_sol(
        self,
        path_to_sol: str,
        write_zeros: bool = True,
    ):
        """
        Write the best feasible primal solution to a file
        """
        best_sol: pyscipopt.scip.Solution = self._model.getBestSol()
        gap: float = self._model.getGap()
        obj_val: float = self._model.getObjVal()

        logger.info(f"Feasible solution found. SCIP objective: {obj_val:.5g} (gap: {gap * 100:.3g}%)")

        try:
            self._model.writeSol(best_sol, path_to_sol, write_zeros=write_zeros)
        except OSError as err:
            logger.error(err)
        else:
            logger.info(f"Sol-file '{path_to_sol}' was successfully written")
