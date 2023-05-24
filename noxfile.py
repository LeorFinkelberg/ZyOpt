import typing as t

import nox
from pathlib2 import Path

# +++++ NOX OPTIONS +++++
nox.needs_version = ">=2022"
nox.options.default_venv_backend = "conda"

# +++++ PROJECT PARAMS +++++
STRATEGY_NAME = "fix_bins_ints_in_relax_sol_with_perturbation"
PROBLEM_FILE_NAME = "model_MNPZ_march_no_plecho_no_CDO_only_BRN.mps"

PATH_TO_DATA_DIR = Path().joinpath("data/").absolute()
PATH_TO_PROBLEM_FILE = PATH_TO_DATA_DIR.joinpath("problems/", PROBLEM_FILE_NAME)
PATH_TO_TARGET_TRAIN_PROBLEM_DIR = PATH_TO_DATA_DIR.joinpath("./problems/train").absolute()
PATH_TO_TARGET_TEST_PROBLEM_DIR = PATH_TO_DATA_DIR.joinpath("./problems/test").absolute()
PATH_TO_MAKE_STRATEGY_FILE = Path("./src/strategy_templates/make_strategy_file.py")
PATH_TO_SETTINGS_DIR = PATH_TO_DATA_DIR.joinpath("settings/")
PATH_TO_RELAX_SET_FILE = PATH_TO_SETTINGS_DIR.joinpath("scip_relax.set")
PATH_TO_MILP_SET_FILE = PATH_TO_SETTINGS_DIR.joinpath("scip_milp.set")
PATH_TO_STRATEGIES_DIR_HOST = PATH_TO_DATA_DIR.joinpath("strategies/")
PATH_TO_STRATEGIES_DIR_CONTAINER = "./data/strategies"
PATH_TO_REQ_FILE = Path("./requirements.txt")

IMAGE_NAME = "tthec"
RELAX_SOLVER_NAME = "scip"
MILP_SOLVER_NAME = "scip"
ROUNDS = 100
DISTRIBUTION_LOWER_THRESHOLD = -0.9
DISTRIBUTION_UPPER_THRESHOLD = 0.1
RANDOM_SEED = 0
FIX_VARS_BINARY = True
FIX_VARS_INTEGER = True
FIX_VARS_DECIMALS = 20
FIX_VARS_LOWER_THRESHOLD = 0.0
FIX_VARS_UPPER_THRESHOLD = 0.0
PARALLEL_N_JOBS_PROBLEMS = -1
PARALLEL_N_JOBS_FEATURES = -1
PARALLEL_PREFER_PROBLEMS = "processes"
PARALLEL_PREFER_FEATURES = "threads"
PARALLEL_VERBOSE_PROBLEMS = 5
PARALLEL_VERBOSE_FEATURES = 5
FEATURES_RELAX_SOL_USE = True
FEATURES_AVG_BIN_THRESHOLDS_USE = True
FEATURES_AVG_BIN_THRESHOLDS_LOWER_THRESHOLD = 0.05
FEATURES_AVG_BIN_THRESHOLDS_UPPER_THRESHOLD = 0.95
FEATURES_AVG_BIN_THRESHOLDS_N_THRESHOLDS = 8
FEATURES_OBJ_COEFF_USE = True
FEATURES_NUMBER_POS_AND_NEG_COEFFS_USE = True
CLUSTERER_MODEL_NAME = "hdbscan"
CLUSTERER_MODEL_MIN_CLUSTER_SIZE = 700
EAD_SUOD_USE = True
EAD_SUOD_COMBINATION = "average"
EAD_SUOD_CONTAMINATION = 0.10
EAD_SUOD_N_JOBS = -1
EAD_SUOD_VERBOSE = True
EAD_COPOD_USE = True
EAD_COPOD_CONTAMINATION = 0.10
EAD_COPOD_N_JOBS = -1
EAD_IFOREST_USE = True
EAD_IFOREST_N_ESTIMATORS = 250
EAD_IFOREST_CONTAMINATION = 0.10
EAD_IFOREST_BEHAVIOUR = "old"
EAD_IFOREST_RANDOM_STATE = 0
EAD_IFOREST_N_JOBS = -1
EAD_HBOSE_USE = True
EAD_HBOSE_N_BINS = 10
EAD_HBOSE_ALPHA = 0.05
EAD_HBOSE_CONTAMINATION = 0.10

# +++++ DOCKER PARAMS +++++
DOCKER_MEMORY = 8000  # Mb
DOCKER_MEMORY_SWAP = 8000  # Mb

# +++++ MISC +++++
DEFAULT_INTERPRETER = "3.8"
TARGET_INTERPRETERS = ("3.8", "3.9", "3.10")

# +++++ ENVS +++++
env = {
    "PYTHONPATH": "./src",
}


@nox.session(python=False)
def run_app_with_docker(session):
    USER_ID = session.run("bash", "-c", "echo $(id -u)", silent=True)
    GROUP_ID = session.run("bash", "-c", "echo $(id -g)", silent=True)
    session.run(
        "sudo", "docker", "build",
        "--build-arg", f"USER_ID={USER_ID}",
        "--build-arg", f"GROUP_ID={GROUP_ID}",
        "--build-arg", f"STRATEGY_NAME={STRATEGY_NAME}",
        "--build-arg", f"PATH_TO_STRATEGIES_DIR={PATH_TO_STRATEGIES_DIR_CONTAINER}",
        "-t",
        IMAGE_NAME,
        ".",
    )

    session.notify("prepare_data_dir")


@nox.session(python=DEFAULT_INTERPRETER)
def prepare_data_dir(session):
    session.install("pathlib2 >= 2.3.7")
    session.run(
        "python", "./src/utils/prepare_data_dir.py",
        "--strategy-name", STRATEGY_NAME,
        "--path-to-problem-file", PATH_TO_PROBLEM_FILE,
        "--path-to-data-dir", PATH_TO_DATA_DIR,
        "--path-to-target-train-problem-dir", PATH_TO_TARGET_TRAIN_PROBLEM_DIR,
        "--path-to-target-test-problem-dir", PATH_TO_TARGET_TEST_PROBLEM_DIR,
        env=env,
    )

    session.notify("make_strategy_file")


@nox.session(python=DEFAULT_INTERPRETER)
def make_strategy_file(session):
    session.install("pathlib2>=2.3.7")
    session.run(
        "python", PATH_TO_MAKE_STRATEGY_FILE,
        "--strategy-name", STRATEGY_NAME,
        "--path-to-test-problem-file", PATH_TO_PROBLEM_FILE,
        "--path-to-relax-set-file", PATH_TO_RELAX_SET_FILE,
        "--path-to-milp-set-file", PATH_TO_MILP_SET_FILE,
        "--path-to-strategies-dir", PATH_TO_STRATEGIES_DIR_HOST,
        "--relax-solver-name", RELAX_SOLVER_NAME,
        "--milp-solver-name", MILP_SOLVER_NAME,
        "--rounds", str(ROUNDS),
        "--distribution-lower-threshold", str(DISTRIBUTION_LOWER_THRESHOLD),
        "--distribution-upper-threshold", str(DISTRIBUTION_UPPER_THRESHOLD),
        "--random-seed", str(RANDOM_SEED),
        "--fix-vars-binary", str(FIX_VARS_BINARY),
        "--fix-vars-integer", str(FIX_VARS_INTEGER),
        "--fix-vars-decimals", str(FIX_VARS_DECIMALS),
        "--fix-vars-lower-threshold", str(FIX_VARS_LOWER_THRESHOLD),
        "--fix-vars-upper-threshold", str(FIX_VARS_UPPER_THRESHOLD),
        "--parallel-n-jobs-problems", str(PARALLEL_N_JOBS_PROBLEMS),
        "--parallel-n-jobs-features", str(PARALLEL_N_JOBS_FEATURES),
        "--parallel-prefer-problems", PARALLEL_PREFER_PROBLEMS,
        "--parallel-prefer-features", PARALLEL_PREFER_FEATURES,
        "--parallel-verbose-problems", str(PARALLEL_VERBOSE_PROBLEMS),
        "--parallel-verbose-features", str(PARALLEL_VERBOSE_FEATURES),
        "--features-relax-sol-use", str(FEATURES_RELAX_SOL_USE),
        "--features-avg-bin-thresholds-use", str(FEATURES_AVG_BIN_THRESHOLDS_USE),
        "--features-avg-bin-thresholds-lower-threshold", str(FEATURES_AVG_BIN_THRESHOLDS_LOWER_THRESHOLD),
        "--features-avg-bin-thresholds-upper-threshold", str(FEATURES_AVG_BIN_THRESHOLDS_UPPER_THRESHOLD),
        "--features-avg-bin-thresholds-n-thresholds", str(FEATURES_AVG_BIN_THRESHOLDS_N_THRESHOLDS),
        "--features-obj-coeff-use", str(FEATURES_OBJ_COEFF_USE),
        "--features-number-pos-and-neg-coeffs-use", str(FEATURES_NUMBER_POS_AND_NEG_COEFFS_USE),
        "--clusterer-model-name", str(CLUSTERER_MODEL_NAME),
        "--clusterer-model-min-cluster-size", str(CLUSTERER_MODEL_MIN_CLUSTER_SIZE),
        "--ead-suod-use", str(EAD_SUOD_USE),
        "--ead-suod-combination", EAD_SUOD_COMBINATION,
        "--ead-suod-contamination", str(EAD_SUOD_CONTAMINATION),
        "--ead-suod-n-jobs", str(EAD_SUOD_N_JOBS),
        "--ead-suod-verbose", str(EAD_SUOD_VERBOSE),
        "--ead-copod-use", str(EAD_COPOD_USE),
        "--ead-copod-contamination", str(EAD_COPOD_CONTAMINATION),
        "--ead-copod-n-jobs", str(EAD_COPOD_N_JOBS),
        "--ead-iforest-use", str(EAD_IFOREST_USE),
        "--ead-iforest-n-estimators", str(EAD_IFOREST_N_ESTIMATORS),
        "--ead-iforest-contamination", str(EAD_IFOREST_CONTAMINATION),
        "--ead-iforest-behaviour", EAD_IFOREST_BEHAVIOUR,
        "--ead-iforest-random-state", str(EAD_IFOREST_RANDOM_STATE),
        "--ead-iforest-n-jobs", str(EAD_IFOREST_N_JOBS),
        "--ead-hbose-use", str(EAD_HBOSE_USE),
        "--ead-hbose-n-bins", str(EAD_HBOSE_N_BINS),
        "--ead-hbose-alpha", str(EAD_HBOSE_ALPHA),
        "--ead-hbose-contamination", str(EAD_HBOSE_CONTAMINATION),
        env=env,
    )
    session.notify("docker_run")


@nox.session(python=False)
def docker_run(session):
    session.run(
        "sudo", "docker", "run",
        "--rm",
        "-v", f"{PATH_TO_DATA_DIR}:/data",
        "-m", f"{DOCKER_MEMORY}m",
        "--memory-swap", f"{DOCKER_MEMORY_SWAP}m",
        IMAGE_NAME,
    )


@nox.session(python=False)
def docker_system_prune(session):
    session.run("sudo", "docker", "system", "prune", "-a")


@nox.session(
    python=TARGET_INTERPRETERS,
    reuse_venv=False,
)
def test(session):
    session.conda_install("pyscipopt==4.3.0", channel="conda-forge")
    session.install("--no-deps", "-r", "requirements.txt")

    session.run(
        "pytest",
        "-v",
        env=env,
    )
