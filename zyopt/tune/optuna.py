import json
import sys
import typing as t
from collections import ChainMap
from operator import attrgetter

import optuna
import pyscipopt
from optuna.study.study import Study
from pathlib2 import Path

from zyopt.common.constants import *
from zyopt.common.exceptions import UnsupportedFileFormatError
from zyopt.common.logger import make_logger
from zyopt.config import INF
from zyopt.strategy import Strategy

logger = make_logger(__file__)

_BOOL_PARAMS: t.Iterable[str] = (NUMERICS_INSTABILITY_KEY,)

_FLAT_PARAMS: t.Iterable[str] = (
    LP_INITALGORITHM_KEY,
    CONFLICT_PREFERBINARY_KEY,
    BRANCHING_PREFERBINARY_KEY,
    HEURISTICS_FARKASDIVING_FREQ_KEY,
    HEURISTICS_FEASPUMP_FREQ_KEY,
    HEURISTICS_RANDROUNDING_FREQ_KEY,
    HEURISTICS_SHIFTANDPROPAGATE_FREQ_KEY,
    HEURISTICS_SHIFTING_FREQ_KEY,
)

_PARAM_NAME_TO_SCIP_PARAMS = {
    LP_INITALGORITHM_KEY: SCIP_PARAM_LP_INITALGORITHM,
    CONFLICT_PREFERBINARY_KEY: SCIP_PARAM_CONFLICT_PREFERBINARY,
    BRANCHING_PREFERBINARY_KEY: SCIP_PARAM_BRANCHING_PREFERBINARY,
    HEURISTICS_FARKASDIVING_FREQ_KEY: SCIP_PARAM_HEURISTICS_FARKASDIVING_FREQ,
    HEURISTICS_FEASPUMP_FREQ_KEY: SCIP_PARAM_HEURISTICS_FEASPUMP_FREQ,
    HEURISTICS_RANDROUNDING_FREQ_KEY: SCIP_PARAM_HEURISTICS_RANDROUNDING_FREQ,
    HEURISTICS_SHIFTANDPROPAGATE_FREQ_KEY: SCIP_PARAM_HEURISTICS_SHIFTANDPROPAGATE_FREQ,
    HEURISTICS_SHIFTING_FREQ_KEY: SCIP_PARAM_HEURISTICS_SHIFTING_FREQ,
    PRESOLVING_DEFAULT_KEY: {},
    PRESOLVING_FAST_KEY: {
        "constraints/varbound/presolpairwise": False,
        "constraints/knapsack/presolpairwise": False,
        "constraints/setppc/presolpairwise": False,
        "constraints/and/presolpairwise": False,
        "constraints/xor/presolpairwise": False,
        "constraints/linear/presolpairwise": False,
        "constraints/logicor/presolpairwise": False,
        "constraints/cumulative/presolpairwise": False,
        "presolving/maxrestarts": 0,
        "propagating/probing/maxprerounds": 0,
        "constraints/components/maxprerounds": 0,
        "presolving/domcol/maxrounds": 0,
        "presolving/gateextraction/maxrounds": 0,
        "presolving/sparsify/maxrounds": 0,
        "presolving/dualsparsify/maxrounds": 0,
        "constraints/logicor/implications": False,
    },
    PRESOLVING_AGGRESSIVE_KEY: {
        "presolving/restartfac": 0.0125,
        "presolving/restartminred": 0.06,
        "constraints/setppc/cliquelifting": True,
        "presolving/boundshift/maxrounds": -1,
        "presolving/qpkktref/maxrounds": -1,
        "presolving/stuffing/maxrounds": -1,
        "presolving/tworowbnd/maxrounds": -1,
        "presolving/dualinfer/maxrounds": -1,
        "presolving/dualagg/maxrounds": -1,
        "presolving/redvub/maxrounds": -1,
        "propagating/probing/maxuseless": 1500,
        "propagating/probing/maxtotaluseless": 75,
    },
    EMPHASIS_FEASIBILITY_KEY: {
        "heuristics/actconsdiving/freq": 20,
        "heuristics/adaptivediving/freq": 3,
        "heuristics/adaptivediving/maxlpiterquot": 0.15,
        "heuristics/bound/freq": 20,
        "heuristics/clique/freq": 20,
        "heuristics/coefdiving/freq": 20,
        "heuristics/completesol/freq": 20,
        "heuristics/conflictdiving/freq": 5,
        "heuristics/conflictdiving/maxlpiterofs": 1500,
        "heuristics/conflictdiving/maxlpiterquot": 0.225,
        "heuristics/crossover/freq": 15,
        "heuristics/dins/freq": 20,
        "heuristics/distributiondiving/freq": 5,
        "heuristics/distributiondiving/maxlpiterofs": 1500,
        "heuristics/distributiondiving/maxlpiterquot": 0.075,
        "heuristics/dps/freq": 20,
        "heuristics/farkasdiving/freq": 5,
        "heuristics/farkasdiving/maxlpiterofs": 1500,
        "heuristics/farkasdiving/maxlpiterquot": 0.075,
        "heuristics/feaspump/freq": 10,
        "heuristics/feaspump/maxlpiterofs": 1500,
        "heuristics/feaspump/maxlpiterquot": 0.015,
        "heuristics/fixandinfer/freq": 20,
        "heuristics/fracdiving/freq": 5,
        "heuristics/fracdiving/maxlpiterofs": 1500,
        "heuristics/fracdiving/maxlpiterquot": 0.075,
        "heuristics/gins/freq": 10,
        "heuristics/guideddiving/freq": 5,
        "heuristics/guideddiving/maxlpiterofs": 1500,
        "heuristics/guideddiving/maxlpiterquot": 0.075,
        "heuristics/zeroobj/freq": 20,
        "heuristics/intdiving/freq": 20,
        "heuristics/intshifting/freq": 5,
        "heuristics/linesearchdiving/freq": 5,
        "heuristics/linesearchdiving/maxlpiterofs": 1500,
        "heuristics/linesearchdiving/maxlpiterquot": 0.075,
        "heuristics/localbranching/freq": 20,
        "heuristics/locks/freq": 20,
        "heuristics/lpface/freq": 8,
        "heuristics/alns/freq": 10,
        "heuristics/nlpdiving/freq": 5,
        "heuristics/mutation/freq": 20,
        "heuristics/multistart/freq": 20,
        "heuristics/mpec/freq": 25,
        "heuristics/objpscostdiving/freq": 10,
        "heuristics/objpscostdiving/maxlpiterofs": 1500,
        "heuristics/objpscostdiving/maxlpiterquot": 0.015,
        "heuristics/octane/freq": 20,
        "heuristics/ofins/freq": 20,
        "heuristics/padm/freq": 20,
        "heuristics/proximity/freq": 20,
        "heuristics/pscostdiving/freq": 5,
        "heuristics/pscostdiving/maxlpiterofs": 1500,
        "heuristics/pscostdiving/maxlpiterquot": 0.075,
        "heuristics/randrounding/freq": 10,
        "heuristics/rens/freq": 20,
        "heuristics/reoptsols/freq": 20,
        "heuristics/repair/freq": 20,
        "heuristics/rins/freq": 13,
        "heuristics/rootsoldiving/freq": 10,
        "heuristics/rootsoldiving/maxlpiterofs": 1500,
        "heuristics/rootsoldiving/maxlpiterquot": 0.015,
        "heuristics/shiftandpropagate/freq": 20,
        "heuristics/shifting/freq": 5,
        "heuristics/trivial/freq": 20,
        "heuristics/trivialnegation/freq": 20,
        "heuristics/trustregion/freq": 20,
        "heuristics/twoopt/freq": 20,
        "heuristics/undercover/freq": 20,
        "heuristics/vbounds/freq": 20,
        "heuristics/veclendiving/freq": 5,
        "heuristics/veclendiving/maxlpiterofs": 1500,
        "heuristics/veclendiving/maxlpiterquot": 0.075,
        "heuristics/rens/nodesofs": 2000,
        "heuristics/rens/minfixingrate": 0.3,
        "heuristics/crossover/nwaitingnodes": 20,
        "heuristics/crossover/dontwaitatroot": True,
        "heuristics/crossover/nodesquot": 0.15,
        "heuristics/crossover/minfixingrate": 0.5,
        "heuristics/alns/trustregion/active": True,
        "heuristics/alns/nodesquot": 0.2,
        "heuristics/alns/nodesofs": 2000,
        "separating/maxrounds": 1,
        "separating/maxroundsroot": 5,
        "separating/aggregation/freq": -1,
        "separating/mcf/freq": -1,
        "nodeselection/restartdfs/stdpriority": 536870911,
    },
    EMPHASIS_OPTIMALITY_KEY: {
        "separating/closecuts/freq": 0,
        "separating/rlt/freq": 20,
        "separating/rlt/maxroundsroot": 15,
        "separating/disjunctive/freq": 20,
        "separating/disjunctive/maxroundsroot": 150,
        "separating/gauge/freq": 0,
        "separating/interminor/freq": 0,
        "separating/convexproj/freq": 0,
        "separating/gomory/maxroundsroot": 15,
        "separating/gomory/maxsepacutsroot": 400,
        "separating/aggregation/maxsepacutsroot": 1000,
        "separating/clique/freq": 20,
        "separating/zerohalf/maxroundsroot": 30,
        "separating/zerohalf/maxsepacutsroot": 200,
        "separating/mcf/freq": 20,
        "separating/mcf/maxsepacutsroot": 400,
        "separating/eccuts/freq": 0,
        "separating/eccuts/maxroundsroot": 375,
        "separating/eccuts/maxsepacutsroot": 100,
        "separating/oddcycle/freq": 0,
        "separating/oddcycle/maxroundsroot": 15,
        "separating/oddcycle/maxsepacutsroot": 10000,
        "constraints/benderslp/sepafreq": 0,
        "constraints/integral/sepafreq": 0,
        "constraints/SOS2/sepafreq": 10,
        "constraints/varbound/sepafreq": 10,
        "constraints/knapsack/sepafreq": 10,
        "constraints/knapsack/maxsepacutsroot": 500,
        "constraints/setppc/sepafreq": 10,
        "constraints/or/sepafreq": 10,
        "constraints/xor/sepafreq": 10,
        "constraints/conjunction/sepafreq": 0,
        "constraints/disjunction/sepafreq": 0,
        "constraints/linear/sepafreq": 10,
        "constraints/linear/maxsepacutsroot": 500,
        "constraints/orbitope/sepafreq": 0,
        "constraints/logicor/sepafreq": 10,
        "constraints/bounddisjunction/sepafreq": 0,
        "constraints/benders/sepafreq": 0,
        "constraints/pseudoboolean/sepafreq": 0,
        "constraints/superindicator/sepafreq": 0,
        "constraints/countsols/sepafreq": 0,
        "constraints/components/sepafreq": 0,
        "cutselection/hybrid/minorthoroot": 0.1,
        "separating/maxroundsrootsubrun": 5,
        "separating/maxaddrounds": 5,
        "separating/maxcutsroot": 5000,
        "constraints/linear/separateall": True,
        "separating/aggregation/maxfailsroot": 200,
        "separating/mcf/maxtestdelta": -1,
        "separating/mcf/trynegscaling": True,
        "branching/fullstrong/maxdepth": 10,
        "branching/fullstrong/priority": 536870911,
        "branching/fullstrong/maxbounddist": 0,
        "branching/relpscost/sbiterquot": 1,
        "branching/relpscost/sbiterofs": 1000000,
        "branching/relpscost/maxreliable": 10,
        "branching/relpscost/usehyptestforreliability": True,
    },
    EMPHASIS_HARDLP_KEY: {
        "heuristics/clique/freq": -1,
        "heuristics/completesol/freq": -1,
        "heuristics/crossover/freq": -1,
        "heuristics/gins/freq": -1,
        "heuristics/locks/freq": -1,
        "heuristics/lpface/freq": -1,
        "heuristics/alns/freq": -1,
        "heuristics/multistart/freq": -1,
        "heuristics/mpec/freq": -1,
        "heuristics/ofins/freq": -1,
        "heuristics/padm/freq": -1,
        "heuristics/rens/freq": -1,
        "heuristics/rins/freq": -1,
        "heuristics/undercover/freq": -1,
        "heuristics/vbounds/freq": -1,
        "heuristics/distributiondiving/freq": -1,
        "heuristics/feaspump/freq": -1,
        "heuristics/fracdiving/freq": -1,
        "heuristics/guideddiving/freq": -1,
        "heuristics/linesearchdiving/freq": -1,
        "heuristics/nlpdiving/freq": -1,
        "heuristics/subnlp/freq": -1,
        "heuristics/objpscostdiving/freq": -1,
        "heuristics/pscostdiving/freq": -1,
        "heuristics/rootsoldiving/freq": -1,
        "heuristics/veclendiving/freq": -1,
        "constraints/varbound/presolpairwise": False,
        "constraints/knapsack/presolpairwise": False,
        "constraints/setppc/presolpairwise": False,
        "constraints/and/presolpairwise": False,
        "constraints/xor/presolpairwise": False,
        "constraints/linear/presolpairwise": False,
        "constraints/logicor/presolpairwise": False,
        "constraints/cumulative/presolpairwise": False,
        "presolving/maxrestarts": 0,
        "propagating/probing/maxprerounds": 0,
        "constraints/components/maxprerounds": 0,
        "presolving/domcol/maxrounds": 0,
        "presolving/gateextraction/maxrounds": 0,
        "presolving/sparsify/maxrounds": 0,
        "presolving/dualsparsify/maxrounds": 0,
        "constraints/logicor/implications": False,
        "branching/relpscost/maxreliable": 1,
        "branching/relpscost/inititer": 10,
        "separating/maxrounds": 1,
        "separating/maxroundsroot": 5,
    },
    EMPHASIS_EASYCIP_KEY: {
        "heuristics/clique/freq": -1,
        "heuristics/completesol/freq": -1,
        "heuristics/locks/freq": -1,
        "heuristics/vbounds/freq": -1,
        "heuristics/rens/freq": -1,
        "heuristics/alns/freq": -1,
        "heuristics/rins/freq": -1,
        "heuristics/gins/freq": -1,
        "heuristics/lpface/freq": -1,
        "heuristics/ofins/freq": -1,
        "heuristics/padm/freq": -1,
        "heuristics/crossover/freq": -1,
        "heuristics/undercover/freq": -1,
        "heuristics/mpec/freq": -1,
        "heuristics/multistart/freq": -1,
        "heuristics/distributiondiving/freq": -1,
        "heuristics/feaspump/freq": -1,
        "heuristics/fracdiving/freq": -1,
        "heuristics/guideddiving/freq": -1,
        "heuristics/linesearchdiving/freq": -1,
        "heuristics/nlpdiving/freq": -1,
        "heuristics/subnlp/freq": -1,
        "heuristics/objpscostdiving/freq": -1,
        "heuristics/pscostdiving/freq": -1,
        "heuristics/rootsoldiving/freq": -1,
        "heuristics/veclendiving/freq": -1,
        "constraints/varbound/presolpairwise": False,
        "constraints/knapsack/presolpairwise": False,
        "constraints/setppc/presolpairwise": False,
        "constraints/and/presolpairwise": False,
        "constraints/xor/presolpairwise": False,
        "constraints/linear/presolpairwise": False,
        "constraints/logicor/presolpairwise": False,
        "constraints/cumulative/presolpairwise": False,
        "presolving/maxrestarts": 0,
        "propagating/probing/maxprerounds": 0,
        "constraints/components/maxprerounds": 0,
        "presolving/domcol/maxrounds": 0,
        "presolving/gateextraction/maxrounds": 0,
        "presolving/sparsify/maxrounds": 0,
        "presolving/dualsparsify/maxrounds": 0,
        "constraints/logicor/implications": False,
        "separating/maxbounddist": 0,
        "constraints/and/sepafreq": 0,
        "separating/aggregation/maxroundsroot": 5,
        "separating/aggregation/maxtriesroot": 100,
        "separating/aggregation/maxaggrsroot": 3,
        "separating/aggregation/maxsepacutsroot": 200,
        "separating/zerohalf/maxsepacutsroot": 200,
        "separating/zerohalf/maxroundsroot": 5,
        "separating/gomory/maxroundsroot": 20,
        "separating/mcf/freq": -1,
    },
    NUMERICS_INSTABILITY_KEY: (
        lambda flag: {
            "numerics/feastol": 1e-05,
            "numerics/dualfeastol": 1e-06,
            "numerics/epsilon": 1e-07,
            "numerics/sumepsilon": 1e-05,
        }
        if flag
        else {}
    ),
}


class ObjectiveParams(t.NamedTuple):
    trial_idx: int
    objective: float
    total_time: float


class Objective:
    """
    Objective function for tuning
    """

    def __init__(
        self,
        path_to_problem: str,
        limits_gap: float = 0.0,
        limits_time: float = INF,
        lpseed: int = 0,
    ):
        # Hold this implementation specific arguments as the fields of the class
        self.path_to_problem = path_to_problem
        self.limits_gap = limits_gap
        self.limits_time = limits_time
        self.lpseed = lpseed

    def __call__(self, trial):
        """
        Calculation an objective value by using the extra arguments
        and return (objective, total_time)
        """
        base_params = {
            SCIP_PARAM_LIMITS_GAP: self.limits_gap,
            SCIP_PARAM_LIMITS_TIME: self.limits_time,
            SCIP_PARAM_RANDOMIZATION_LPSEED: self.lpseed,
            SCIP_PARAM_CONFLICT_PREFERBINARY: trial.suggest_categorical(
                CONFLICT_PREFERBINARY_KEY, [True, False]
            ),
            SCIP_PARAM_BRANCHING_PREFERBINARY: trial.suggest_categorical(
                BRANCHING_PREFERBINARY_KEY, [True, False]
            ),
            SCIP_PARAM_HEURISTICS_FARKASDIVING_FREQ: trial.suggest_categorical(
                HEURISTICS_FARKASDIVING_FREQ_KEY, [-1, 10]
            ),
            SCIP_PARAM_HEURISTICS_FEASPUMP_FREQ: trial.suggest_categorical(
                HEURISTICS_FEASPUMP_FREQ_KEY, [-1, 20]
            ),
            SCIP_PARAM_HEURISTICS_RANDROUNDING_FREQ: trial.suggest_categorical(
                HEURISTICS_RANDROUNDING_FREQ_KEY, [-1, 20]
            ),
            SCIP_PARAM_HEURISTICS_SHIFTANDPROPAGATE_FREQ: trial.suggest_categorical(
                HEURISTICS_SHIFTANDPROPAGATE_FREQ_KEY, [-1, 20]
            ),
            SCIP_PARAM_HEURISTICS_SHIFTING_FREQ: trial.suggest_categorical(
                HEURISTICS_SHIFTING_FREQ_KEY, [-1, 10]
            ),
        }

        # Presolving
        presolving_emphasis = trial.suggest_categorical(
            "presolving_emphasis",
            [
                PRESOLVING_DEFAULT_KEY,
                PRESOLVING_AGGRESSIVE_KEY,
                PRESOLVING_FAST_KEY,
            ],
        )
        presolving_params = _PARAM_NAME_TO_SCIP_PARAMS.get(presolving_emphasis)
        base_params.update(presolving_params)

        # Init algorithm
        lp_initalgorithm = trial.suggest_categorical(LP_INITALGORITHM_KEY, ["s", "p", "d", "b"])
        base_params.update({SCIP_PARAM_LP_INITALGORITHM: lp_initalgorithm})

        # Emphasis
        emphasis = trial.suggest_categorical(
            "emphasis",
            [
                EMPHASIS_EASYCIP_KEY,
                EMPHASIS_FEASIBILITY_KEY,
                EMPHASIS_OPTIMALITY_KEY,
                EMPHASIS_HARDLP_KEY,
            ],
        )
        emphasis_params = _PARAM_NAME_TO_SCIP_PARAMS.get(emphasis)
        base_params.update(emphasis_params)

        # Numerics feas
        numerics_instability = trial.suggest_categorical("numerics_instability", [True, False])
        numerics_params = _PARAM_NAME_TO_SCIP_PARAMS.get(NUMERICS_INSTABILITY_KEY)(numerics_instability)
        base_params.update(numerics_params)

        model = pyscipopt.Model()
        model.readProblem(self.path_to_problem)
        logger.info(f"Reading params trial-{trial.number} ...")
        model.setParams(base_params)

        model.optimize()
        status = model.getStatus()

        if (
            status
            in (
                SCIP_STATUS_TIMELIMIT,
                SCIP_STATUS_GAPLIMIT,
                SCIP_STATUS_OPTIMAL,
            )
            and model.getSols()
        ):
            return model.getObjVal(), model.getTotalTime()
        elif status == SCIP_STATUS_INFEASIBLE:
            return float("inf"), float("inf")


class SolverParamsTuner(Strategy):
    """
    Tuner solver params

    Example:
        ```python
        tuner = SolverParamsTuner(
            study_name="scip_tune",
            storage="sqlite:///scip_tune.db",
            path_to_problem="./data/problems/problem.mps",
            n_trials=150,
            limits_time=600,  # 600 sec
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
            direction_for_objective="maximize",
            show_progress_bar=True,
        )
        study = tuner.tune()
        objective, total_time = study.best_trials[-1].values
        print(study.best_trials[-1].params)
        ```
        ```python
        tuner = SolverParamsTuner(
            study_name="scip_tune",
            path_to_problem="./data/problems/problem.mps",
            limits_time=600,  # 600 sec
            n_jobs=-1,
            limits_gap=0.01,
            limits_time=60,
            show_progress_bar=True,
        )
        best_params: dict = tuner.tuner(return_best_params_by_time=True)
        model = pyscipopt.Model()
        model.readProblem("./data/problems/other_problem.mps")
        model.setParams(best_params)
        model.optimize()
        status = model.getStatus()
        ```
    """

    def __init__(
        self,
        study_name: str,
        path_to_problem: str,
        n_trials: int = 100,
        optimize_timeout: t.Optional[float] = None,
        n_jobs: int = 1,
        catch: t.Union[t.Iterable[t.Type[Exception]], t.Type[Exception]] = (),
        callbacks: t.Optional[
            t.List[t.Callable[[Study, optuna.trial._frozen.FrozenTrial], None]]
        ] = None,
        limits_gap: t.Optional[float] = 0.01,
        limits_time: t.Optional[float] = 180,
        lpseed: int = 0,
        sampler: t.Optional[optuna.samplers._base.BaseSampler] = None,
        pruner: t.Optional[optuna.pruners._base.BasePruner] = None,
        storage: t.Optional[str] = None,
        load_if_exists: bool = False,
        direction_for_objective: str = "minimize",
        direction_for_time: str = "minimize",
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ):
        self.study_name = study_name
        self.storage = storage
        self.path_to_problem = path_to_problem
        self.sampler = sampler
        self.pruner = pruner
        self.load_if_exists = load_if_exists
        self.n_trials = n_trials
        self.optimize_timeout = optimize_timeout
        self.n_jobs = n_jobs
        self.catch = catch
        self.callbacks = callbacks
        self.limits_gap = limits_gap
        self.limits_time = limits_time
        self.lpseed = lpseed
        self.direction_for_objective = direction_for_objective
        self.direction_for_time = direction_for_time
        self.gc_after_trial = gc_after_trial
        self.show_progress_bar = show_progress_bar
        self.obj = Objective(
            path_to_problem=self.path_to_problem,
            limits_gap=self.limits_gap,
            limits_time=self.limits_time,
            lpseed=self.lpseed,
        )

    def optimize(self, return_best_params_by_time: bool = False) -> t.Union[Study, dict]:
        """
        Tune solver params
        """
        study = optuna.create_study(
            directions=(self.direction_for_objective, self.direction_for_time),
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=self.load_if_exists,
        )
        study.optimize(
            func=self.obj,
            n_trials=self.n_trials,
            timeout=self.optimize_timeout,
            n_jobs=self.n_jobs,
            catch=self.catch,
            callbacks=self.callbacks,
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar,
        )

        if return_best_params_by_time:
            _chain: t.List[dict] = []
            best_params_trial_idx: int = min(
                (
                    ObjectiveParams(trial_idx, *study.best_trials[trial_idx].values)
                    for trial_idx in range(len(study.best_trials))
                ),
                key=attrgetter("total_time"),
            ).trial_idx

            _params: dict = study.best_trials[best_params_trial_idx].params

            for key, value in _params.items():
                if key in _FLAT_PARAMS:
                    _sub_params = {_PARAM_NAME_TO_SCIP_PARAMS.get(key): _params.get(key)}
                elif key in _BOOL_PARAMS:
                    _sub_params = _PARAM_NAME_TO_SCIP_PARAMS.get(key)(value)
                else:
                    _sub_params = _PARAM_NAME_TO_SCIP_PARAMS.get(value)

                _chain.append(_sub_params)

            self.best_params = dict(ChainMap(*_chain))

            return self.best_params
        else:
            if study.best_trials:
                return study
            else:
                logger.info(PROCESS_INTERRUPT_MSG)
                sys.exit(-1)

    def write_best_params(self, path_to_settings: str):
        """
        Write best params in JSON or SCIP formats
        """
        suffix = Path(path_to_settings).suffix
        try:
            with open(path_to_settings, mode="w") as f:
                if suffix == ".set":
                    for key, value in self.best_params.items():
                        f.write(f"{key} = {value}\n")
                elif suffix == ".json":
                    json.dump(self.best_params, f)
                else:
                    raise UnsupportedFileFormatError(
                        f"Error! Unsupported file format: {suffix}. Valid formats: {self._valid_formats}"
                    )
        except OSError:
            raise
        else:
            logger.info(FILE_SUCCESS_WRITE_MSG.format(path_to_settings))

    def _validate_params(self):
        """
        Validate params
        """

        self._valid_formats: t.Set[str] = {".json", ".set"}
