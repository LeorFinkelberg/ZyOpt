import sys
import typing as t

import pandas as pd
import pyscipopt
from pathlib2 import Path
from tqdm import tqdm

from zyopt.common.constants import *
from zyopt.common.logger import make_logger
from zyopt.config import PYSCIPOPT_APACHE_2_0_LICENSE_VERSION
from zyopt.strategy import Strategy

logger = make_logger(__file__)

PYSCIPOPT_CURRENT_VERSION = tuple(map(int, pyscipopt.__version__.split(".")))
if PYSCIPOPT_CURRENT_VERSION < PYSCIPOPT_APACHE_2_0_LICENSE_VERSION:
    logger.warning(
        f"You are using SCIP version {'.'.join(map(str, PYSCIPOPT_CURRENT_VERSION))}, "
        "which is only available under ZIB ACADEMIC LICENSE. \n\t"
        "See https://www.scipopt.org/academic.txt"
    )


class Scip(Strategy):
    """
    Simple wrapper for SCIP solver

    Example:
        model = Scip(
            solver_mode="milp",  # "relax" or "milp"
            path_to_problem="./data/problems/problem.mps",
            path_to_params="./data/settings/scip_milp.set",
        )
        model.optimize()
        status = model.get_status()
        ...
        # Or
        other_model = pyscipopt.Model()
        other_model.readProblem("...")
        other_model.readParams("...")

        model = Scip(
            solver_mode="milp",
            path_to_params="./data/problems/problem.mps",
            problem=other_model,
        )
    """

    def __init__(
        self,
        *,
        solver_mode: str,
        path_to_params: str,
        path_to_problem: t.Optional[str] = None,
        problem: t.Optional[pyscipopt.scip.Model] = None,
    ):
        self.solver_mode = solver_mode
        self.path_to_params = Path(path_to_params)

        if problem is not None:
            logger.info(f"Reading problem: {problem}")
            self._model = problem
            self.path_to_problem = None
        else:
            self.path_to_problem = Path(path_to_problem)
            self._model = pyscipopt.Model()
            logger.info(f"Reading problem: {self.path_to_problem}")
            self._model.readProblem(self.path_to_problem)

        logger.info(f"Reading params: {self.path_to_params}")
        self._model.readParams(self.path_to_params)

        self.all_vars: t.Iterable[pyscipopt.scip.Variable] = self._model.getVars()
        self.all_var_names: t.Iterable[str] = [var.name for var in self.all_vars]
        self._all_vars = pd.Series(self.all_vars, index=self.all_var_names)

        # Validate params
        self._validate_params()

    def optimize(self):
        """
        Optimize the problem
        """
        # We have to extract variables again to break the connection
        # of the model with the attributes of the class
        _all_vars: t.Iterable[pyscipopt.scip.Variable] = self._model.getVars()

        if self.solver_mode == SOLVER_MODE_RELAX:
            _bin_int_vars: t.List[pyscipopt.scip.Variable] = [
                var for var in _all_vars if var.vtype() != VAR_TYPE_CONTINUOUS
            ]

            var: pyscipopt.scip.Variable
            for var in tqdm(_bin_int_vars):
                self._model.chgVarType(var, VAR_TYPE_CONTINUOUS)

        logger.info(f"Running SCIP in {self.solver_mode.upper()} mode ...")
        self._model.optimize()

    def convert_best_sol_to_dict(self) -> dict:
        """
        Converts SCIP solution to dict
        """
        sol = self._model.getBestSol()
        return {var.name: self._model.getSolVal(sol, var) for var in self.all_vars}

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

        if status in (
            SCIP_STATUS_OPTIMAL,
            SCIP_STATUS_GAPLIMIT,
            SCIP_STATUS_TIMELIMIT,
            SCIP_STATUS_USERINTERRUPT,
        ):
            if self.get_sols():
                return self.convert_best_sol_to_dict()
            else:
                logger.info(PROCESS_INTERRUPT_MSG)
                sys.exit(-1)
        elif status == SCIP_STATUS_INFEASIBLE:
            return None

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

    def fix_vars(self, base_for_fix: pd.Series, var_names: t.List[str]) -> pyscipopt.scip.Model:
        """
        Fix vars in base
        """
        for var in tqdm(self._all_vars.loc[var_names]):
            var_name: str = var.name
            value: float = base_for_fix.loc[var_name]
            self._model.fixVar(var, value)

        return self._model

    def _validate_params(self):
        """
        Validate params
        """
        self._valid_solver_mode: t.Set[str] = {SOLVER_MODE_RELAX, SOLVER_MODE_MILP}

        if self.solver_mode not in self._valid_solver_mode:
            raise ValueError(
                f"Error! Unknown solver mode: {self.solver_mode}. "
                f"Valid solver mode: {list(self._valid_solver_mode)}"
            )

        self._check_file_path(self.path_to_problem)
        self._check_file_path(self.path_to_params)
