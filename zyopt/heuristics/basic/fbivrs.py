import typing as t

import numpy as np
import pandas as pd
import pyscipopt

from zyopt.base_model import Model
from zyopt.common.constants import *
from zyopt.common.logger import make_logger
from zyopt.config import DECIMALS, RANDOM_SEED
from zyopt.default.scip import ScipModel

logger = make_logger(__file__)


class FixBinaryIntegerVarsInRelaxSolModel(Model):
    """
    Fix Binary and Integer Variables in Relax Solution with Perturbation
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

    def optimize(self):
        self._relax_model = ScipModel(
            solver_mode=SOLVER_MODE_RELAX,
            path_to_problem=self.path_to_problem,
            path_to_params=self.path_to_relax_params,
        )
        self._relax_model.optimize()
        sol: t.Optional[dict] = self._relax_model.get_best_sol()

        self._milp_model = ScipModel(
            solver_mode=SOLVER_MODE_MILP,
            path_to_problem=self.path_to_problem,
            path_to_params=self.path_to_milp_params,
        )

        if sol is not None:
            _relax_sol = pd.Series(sol)
            relax_sol = _relax_sol.loc[
                self._get_var_names(self.use_binary, VAR_TYPE_BINARY)
                + self._get_var_names(self.use_integer, VAR_TYPE_INTEGER)
            ]
            relax_sol = np.round(relax_sol, decimals=DECIMALS)

            mask = self._make_mask(
                relax_sol, self.fix_vars_lower_threshold, self.fix_vars_upper_threshold
            )
            var_names: t.List[str] = relax_sol.loc[mask].index.to_list()

            _modif_model = self._milp_model.fix_vars(relax_sol, var_names)
            _milp_model = ScipModel(
                solver_mode=SOLVER_MODE_MILP,
                path_to_params=self.path_to_milp_params,
                model=_modif_model,
            )
            _milp_model.optimize()
            status = _milp_model.get_status()

            if status == SCIP_STATUS_IFEASIBLE:
                np.random.seed(RANDOM_SEED)

                for round_ in range(self.perturb_rounds):
                    logger.info(
                        f"Running SCIP with perturbation relax sol: {round_ + 1} / {self.perturb_rounds}"
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
                        relax_sol_with_noise,
                        self.fix_vars_lower_threshold,
                        self.fix_vars_upper_threshold,
                    )
                    var_names: t.List[str] = relax_sol_with_noise.loc[mask].index.to_list()

                    _milp_model = ScipModel(
                        solver_mode=SOLVER_MODE_MILP,
                        path_to_problem=self.path_to_problem,
                        path_to_params=self.path_to_milp_params,
                    )
                    _modif_model = _milp_model.fix_vars(relax_sol_with_noise, var_names)
                    _milp_model = ScipModel(
                        solver_mode=SOLVER_MODE_MILP,
                        path_to_params=self.path_to_milp_params,
                        model=_modif_model,
                    )
                    _milp_model.optimize()
                    status = _milp_model.get_status()

                    if status == SCIP_STATUS_IFEASIBLE:
                        continue
                    else:
                        _milp_model.write_best_sol(self.path_to_sol)
                        break
            else:
                _milp_model.write_best_sol(self.path_to_sol)

        else:
            logger.info("Infeasible RELAX solution. Check settings")

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
                vars_=self._milp_model.all_vars_,
                var_type=var_type,
            )
            if use_vars
            else []
        )

        return [var.name for var in vars_]
