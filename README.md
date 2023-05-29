# _ZyOpt_

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ZyOpt is add-in for the SCIP solver with support for:
- heuristics (for example, fixing zero binary and integer variables)
- based on machine (for example, detection of zero binary variables as anomalies) and deep learning algorithms (for example, GCNN)

### _How to use_

#### _Simple SCIP_
```python
from zyopt.default.scip import Scip

model = Scip(
    solver_mode="milp",
    path_to_problem="./data/problems/problem.mps",
    path_to_params="./data/settings/scip_milp.set",
)
model.optimize()
model.write_best_sol("./data/sols/milp.sol")
```

#### _Fixation binaries and integers variables in relax sol with perturbation_
```python
from zyopt.heuristics.basic.fbivrs import InRelaxSolVarsFixator

model = InRelaxSolVarsFixator(
    path_to_problem="./data/problems/problem.mps",
    path_to_relax_params="./data/settings/scip_relax.set",
    path_to_milp_params="./data/settings/scip_milp.set",
    use_binary=False,
    use_integer=True,
)
model.optimize()
model.write_best_sol("./data/sols/milp.sol")
```

#### _Tune hyperparams for SCIP_
```python
from zyopt.tune.optuna import SolverParamsTuner
from zyopt.default.scip import Scip

tuner = SolverParamsTuner(
    study_name="scip_tune",
    storage="sqlite:///scip_tuner.db",
    path_to_problem="./data/problems/problem.mps",
    n_trials=350,
    optimize_timeout=600,  # 600 sec
    show_progress_bar=True,
)
best_params: dict = tuner.optimize(return_best_params_by_time=True)
# tuner.write_best_params("./best_params.json")
tuner.write_best_params("./data/settings/best_params.set")

model = Scip(
    solver_mode="milp",
    path_to_problem="./data/problems/problem.mps",
    path_to_params="./data/settings/best_params.set",
)
model.optimize()
model.write_best_sol("./data/sols/milp.sol")

```
