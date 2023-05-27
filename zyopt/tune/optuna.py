import typing as t

import numpy as np
import optuna
import pyscipopt

from zyopt.common.constants import *


class Objective:
    """
    Objective function for tuning
    """

    def __init__(self, path_to_problem: str):
        # Hold this implementation specific arguments as the fields of the class
        self.path_to_problem = path_to_problem

    def __call__(self, trial):
        # Calculation an objective value by using the extra arguments
        base_params = {
            "conflict/preferbinary": trial.suggest_categorical("conflict_preferbinary", [True, False]),
            "branching/preferbinary": trial.suggest_categorical("branching_preferbinary", [True, False]),
            "heuristics/farkasdiving/freq": trial.suggest_categorical(
                "heuristics_farkasdiving_freq", [-1, 10]
            ),
            "heuristics/feaspump/freq": trial.suggest_categorical("heuristics_feaspump_freq", [-1, 20]),
            "heuristics/randrounding/freq": trial.suggest_categorical(
                "heuristics_randrounding_freq", [-1, 20]
            ),
            "heuristics/shiftandpropagate/freq": trial.suggest_categorical(
                "heuristics_shiftandpropagate_freq", [-1, 20]
            ),
            "heuristics/shifting/freq": trial.suggest_categorical("heuristics_shifting_freq", [-1, 10]),
        }

        # Presolving
        presolving_emphasis = trial.suggest_categorical(
            "presolving_emphasis",
            [
                PRESOLVING_DEFAULT,
                PRESOLVING_AGGRESSIVE,
                PRESOLVING_FAST,
            ],
        )

        if presolving_emphasis == PRESOLVING_AGGRESSIVE:
            presolving_params = {
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
            }
        elif presolving_emphasis == PRESOLVING_FAST:
            presolving_params = {
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
            }
        else:
            # default
            presolving_params = {}

        base_params.update(presolving_params)

        # Init algorithm
        base_params.update(
            {
                "lp/initalgorithm": trial.suggest_categorical("lp_initalgorithm", ["s", "p", "d", "b"]),
                "randomization/lpseed": 0,
            }
        )

        # Emphasis
        emphasis = trial.suggest_categorical(
            "emphasis", [EMPHASIS_EASYCIP, EMPHASIS_FEASIBILITY, EMPHASIS_OPTIMALITY, EMPHASIS_HARDLP]
        )

        if emphasis == EMPHASIS_FEASIBILITY:
            emphasis_params = {
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
            }
        elif emphasis == EMPHASIS_OPTIMALITY:
            emphasis_params = {
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
            }
        elif emphasis == EMPHASIS_HARDLP:
            emphasis_params = {
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
            }
        else:
            # easycip
            emphasis_params = {
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
            }

        base_params.update(emphasis_params)

        # Numerics feas
        numerics_instability = trial.suggest_categorical("numerics_instability", [True, False])
        numerics_params = (
            {
                "numerics/feastol": 1e-05,
                "numerics/dualfeastol": 1e-06,
                "numerics/epsilon": 1e-07,
                "numerics/sumepsilon": 1e-05,
            }
            if numerics_instability
            else {}
        )

        base_params.update(numerics_params)

        model = pyscipopt.Model()
        model.readProblem(self.path_to_problem)
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
            return model.getObjVal()
        elif status == SCIP_STATUS_INFEASIBLE:
            return np.float("inf")


class SolverParamsTuner:
    def __init__(
        self,
        study_name: str,
        path_to_problem: str,
        n_trials: int = 100,
        sampler: t.Optional[optuna.samplers._base.BaseSampler] = None,
        pruner: t.Optional[optuna.pruners._base.BasePruner] = None,
        path_to_storage: t.Optional[str] = None,
        load_if_exsits: bool = False,
        direction: str = "minimize",
    ):
        self.study_name = study_name
        self.path_to_storage = f"sqlite:///{path_to_storage}"
        self.path_to_problem = path_to_problem
        self.sampler = sampler
        self.pruner = pruner
        self.load_if_exists = load_if_exsits
        self.n_trials = n_trials
        self.direction = direction
        self.obj = Objective(self.path_to_problem)

    def tune(self) -> dict:
        """
        Tune solver params
        """
        study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.path_to_storage,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=self.load_if_exists,
        )
        study.optimize(Objective(self.path_to_problem), n_trials=self.n_trials)

        return study.best_params
