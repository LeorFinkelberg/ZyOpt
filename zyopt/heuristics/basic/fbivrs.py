import sys
import typing as t

import numpy as np
import pandas as pd
import pyscipopt

from zyopt.common.auxiliary_functions import timer
from zyopt.common.constants import *
from zyopt.common.logger import make_logger
from zyopt.config import DECIMALS, RANDOM_SEED
from zyopt.default.scip import Scip
from zyopt.strategy import Strategy

logger = make_logger(__file__)


class InRelaxSolVarsFixator(Strategy):
    """
    Fix Binary and Integer Variables in Relax Solution with Perturbation

    Example:
        ```python
        model = InRelaxSolVarsFixator(
            path_to_problem="./data/problems/problem.mps",
            path_to_relax_params="./data/settings/scip_relax.set",
            path_to_milp_params="./data/settings/scip_milp.set",
            path_to_sol="./data/sols/milp.sol",
            use_binary=False,
            use_integer=True,
            fix_vars_lower_threshold: float = 0.0,
            fix_vars_upper_threshold: float = 0.0,
            perturb_rounds: int = 350,
            distr_lower_threshold: float = -0.9,
            distr_upper_threshold: float = 0.1,
        )
        model.optimize()
        ```
    """

    def __init__(
        self,
        path_to_problem: str,
        path_to_relax_params: str,
        path_to_milp_params: str,
        path_to_sol: str,
        use_binary: bool = True,
        use_integer: bool = True,
        fix_vars_lower_threshold: float = 0.0,
        fix_vars_upper_threshold: float = 0.0,
        perturb_rounds: int = 100,
        distr_lower_threshold: float = -0.9,
        distr_upper_threshold: float = 0.1,
    ):
        self.strategy_name = self.__class__.__name__.upper()
        self.path_to_problem = path_to_problem
        self.path_to_relax_params = path_to_relax_params
        self.path_to_milp_params = path_to_milp_params
        self.path_to_sol = path_to_sol
        self.use_binary = use_binary
        self.use_integer = use_integer
        self.fix_vars_lower_threshold = fix_vars_lower_threshold
        self.fix_vars_upper_threshold = fix_vars_upper_threshold
        self.perturb_rounds = perturb_rounds
        self.distr_lower_threshold = distr_lower_threshold
        self.distr_upper_threshold = distr_upper_threshold

        # Validate params
        self._validate_params()

    @timer
    def optimize(self):
        """
        Optimize the problem
        """
        _relax_model = Scip(
            solver_mode=SOLVER_MODE_RELAX,
            path_to_problem=self.path_to_problem,
            path_to_params=self.path_to_relax_params,
        )
        _relax_model.optimize()
        _relax_sol: t.Optional[dict] = _relax_model.get_best_sol()
        _milp_model: Scip = self._reset_model()
        # Attribute `all_vars` has to be created outside
        # of `__init__` method to avoid the SEGMENTATION FAULT error
        self.all_vars: t.Iterable[pyscipopt.scip.Variable] = _milp_model.all_vars

        if _relax_sol is not None:
            relax_sol: pd.Series = self._prepare_sol(_relax_sol)

            mask = self._make_mask(
                base_for_fix=relax_sol,
                lower_threshold=self.fix_vars_lower_threshold,
                upper_threshold=self.fix_vars_upper_threshold,
            )
            var_names: t.List[str] = relax_sol.loc[mask].index.to_list()

            _modif_problem: pyscipopt.scip.Model = _milp_model.fix_vars(
                base_for_fix=relax_sol, var_names=var_names
            )
            _milp_model = Scip(
                solver_mode=SOLVER_MODE_MILP,
                path_to_params=self.path_to_milp_params,
                problem=_modif_problem,
            )
            logger.info(f"Running {self.strategy_name} without perturbation")
            _milp_model.optimize()
            status: str = _milp_model.get_status()

            if status in (
                SCIP_STATUS_OPTIMAL,
                SCIP_STATUS_GAPLIMIT,
                SCIP_STATUS_TIMELIMIT,
                SCIP_STATUS_USERINTERRUPT,
            ):
                if _milp_model.get_sols():
                    _milp_model.write_best_sol(self.path_to_sol)
                else:
                    logger.info(PROCESS_INTERRUPT_MSG)
                    sys.exit(-1)
            elif status == SCIP_STATUS_INFEASIBLE:
                np.random.seed(RANDOM_SEED)

                for round_ in range(self.perturb_rounds):
                    logger.info(
                        f"Running {self.strategy_name} with perturbation relax sol: "
                        f"{round_ + 1} / {self.perturb_rounds}"
                    )

                    noise = np.random.uniform(
                        low=self.distr_lower_threshold,
                        high=self.distr_upper_threshold,
                        size=relax_sol.shape[0],
                    )
                    relax_sol_with_noise = relax_sol + noise
                    relax_sol_with_noise = pd.Series(
                        np.where(relax_sol_with_noise < 0.0, 0.0, relax_sol_with_noise),
                        index=relax_sol.index,
                    )
                    mask = self._make_mask(
                        base_for_fix=relax_sol_with_noise,
                        lower_threshold=self.fix_vars_lower_threshold,
                        upper_threshold=self.fix_vars_upper_threshold,
                    )
                    var_names: t.List[str] = relax_sol_with_noise.loc[mask].index.to_list()

                    _milp_model: Scip = self._reset_model()
                    _modif_problem = _milp_model.fix_vars(relax_sol_with_noise, var_names)
                    _milp_model = Scip(
                        solver_mode=SOLVER_MODE_MILP,
                        path_to_params=self.path_to_milp_params,
                        problem=_modif_problem,
                    )
                    _milp_model.optimize()
                    status: str = _milp_model.get_status()

                    if status == SCIP_STATUS_INFEASIBLE:
                        logger.info(
                            "Unfortunately, it was not possible to find a solution. "
                            "We try another perturbation of the relaxed solution"
                        )
                        continue
                    elif status in (
                        SCIP_STATUS_OPTIMAL,
                        SCIP_STATUS_GAPLIMIT,
                        SCIP_STATUS_TIMELIMIT,
                        SCIP_STATUS_USERINTERRUPT,
                    ):
                        if _milp_model.get_sols():
                            _milp_model.write_best_sol(self.path_to_sol)
                            break
                        elif status == SCIP_STATUS_USERINTERRUPT:
                            logger.info(PROCESS_INTERRUPT_MSG)
                            sys.exit(-1)
        else:
            logger.info("Infeasible RELAX solution...")
            sys.exit(-1)

    @staticmethod
    def _get_vars_by_type(
        vars_: t.Iterable[pyscipopt.scip.Variable],
        var_type: str,
    ) -> t.Iterable[pyscipopt.scip.Variable]:
        """
        Gets vars by type
        """
        return [var for var in vars_ if var.vtype() == var_type]

    def _get_var_names(self, use_vars: bool, var_type: str) -> t.List[str]:
        """
        Get var names
        """
        vars_: t.List[pyscipopt.scip.Variable] = (
            self._get_vars_by_type(
                vars_=self.all_vars,
                var_type=var_type,
            )
            if use_vars
            else []
        )

        return [var.name for var in vars_]

    def _prepare_sol(self, sol: dict) -> pd.Series:
        """
        Prepare solution
        """
        _relax_sol = pd.Series(sol)
        relax_sol = _relax_sol.loc[
            self._get_var_names(self.use_binary, VAR_TYPE_BINARY)
            + self._get_var_names(self.use_integer, VAR_TYPE_INTEGER)
        ]
        return np.round(relax_sol, decimals=DECIMALS)

    def _reset_model(self) -> Scip:
        """
        Reset model
        """
        return Scip(
            solver_mode=SOLVER_MODE_MILP,
            path_to_problem=self.path_to_problem,
            path_to_params=self.path_to_milp_params,
        )

    def _validate_params(self):
        """
        Validate params
        """

        self._check_file_path(self.path_to_problem)
        self._check_file_path(self.path_to_relax_params)
        self._check_file_path(self.path_to_milp_params)

        for param in (self.use_binary, self.use_integer):
            if not isinstance(param, bool):
                raise ValueError(f"Error! Flags must be bool values, but: {param}")

        if (0 > self.perturb_rounds) or (self.perturb_rounds > 1_000_000):
            raise ValueError(
                "Error! Param `perturb_rounds` must belong to range [0, 1_000_000], "
                f"but: {self.perturb_rounds}"
            )
