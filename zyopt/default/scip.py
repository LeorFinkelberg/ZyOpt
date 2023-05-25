import sys
import typing as t

import pandas as pd
import pyscipopt
from pathlib2 import Path
from tqdm import tqdm

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
        solver_mode: str,
        path_to_params: str,
        path_to_problem: t.Optional[str] = None,
        model: t.Optional[pyscipopt.scip.Model] = None,
    ):
        PYSCIPOPT_CURRENT_VERSION = tuple(map(int, pyscipopt.__version__.split(".")))
        if PYSCIPOPT_CURRENT_VERSION < PYSCIPOPT_APACHE_2_0_LICENSE_VERSION:
            logger.warning(
                f"You are using SCIP version {'.'.join(map(str, PYSCIPOPT_CURRENT_VERSION))}, "
                "which is only available under ZIB ACADEMIC LICENSE. "
                "See https://www.scipopt.org/academic.txt"
            )
        self.solver_mode = solver_mode
        self.path_to_params = Path(path_to_params)

        if model is not None:
            self._model = model
        else:
            try:
                self.path_to_problem = Path(path_to_problem)
            except TypeError as err:
                raise

            self._model = pyscipopt.Model()
            logger.info(f"Reading problem: {self.path_to_problem}")
            self._model.readProblem(self.path_to_problem)

        logger.info(f"Reading params: {self.path_to_params}")
        self._model.readParams(self.path_to_params)

        self.all_vars_: t.Iterable[pyscipopt.scip.Variable] = self._model.getVars()
        self.all_var_names_: t.Iterable[str] = [var.name for var in self.all_vars_]
        self.all_vars = pd.Series(self.all_vars_, index=self.all_var_names_)

    def optimize(self):
        """
        Optimize the problem
        """
        _all_vars: t.Iterable[pyscipopt.scip.Variable] = self._model.getVars()

        if self.solver_mode == SOLVER_MODE_RELAX:
            _bin_int_vars: t.List[pyscipopt.scip.Variable] = [
                var for var in _all_vars if var.vtype() != VAR_TYPE_CONTINUOUS
            ]

            var: pyscipopt.scip.Variable
            for var in _bin_int_vars:
                self._model.chgVarType(var, VAR_TYPE_CONTINUOUS)

        logger.info(f"Running SCIP in {self.solver_mode.upper()} mode ...")
        self._model.optimize()

    def convert_best_sol_to_dict(self) -> dict:
        """
        Converts SCIP solution to dict
        """
        sol = self._model.getBestSol()
        return {var.name: self._model.getSolVal(sol, var) for var in self.all_vars_}

    def get_sols(self) -> t.Iterable[pyscipopt.scip.Solution]:
        """
        Retrieve list of all feasible primal solutions stored in the solution storage
        """
        return self._model.getSols()

    def get_best_sol(self) -> t.Optional[dict]:
        """
        Get the best SCIP solution
        """
        status: str = self.get_status()

        if status == SCIP_STATUS_IFEASIBLE:
            return None
        elif status in (
            SCIP_STATUS_OPTIMAL,
            SCIP_STATUS_GAPLIMIT,
            SCIP_STATUS_TIMELIMIT,
            SCIP_STATUS_USERINTERRUPT,
        ):
            pool_sols: t.Iterable[pyscipopt.scip.Solution] = self.get_sols()

            if pool_sols:
                return self.convert_best_sol_to_dict()
            else:
                logger.info("Process is interrupted ...")
                sys.exit(-1)

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

    def get_vars(self) -> t.Iterable[pyscipopt.scip.Variable]:
        """
        Retrive all variables
        """
        return self._model.getVars()

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

    def fix_vars(self, base_for_fix: pd.Series, var_names: t.List[str]):
        """
        Fix vars in base
        """
        for var in tqdm(self.all_vars.loc[var_names]):
            var_name: str = var.name
            value: float = base_for_fix.loc[var_name]
            self._model.fixVar(var, value)

        return self._model
