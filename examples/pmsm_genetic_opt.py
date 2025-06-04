import json
from pathlib import Path
from deap import base, creator, tools, algorithms
import argparse
import random

# Path to default machine JSON
DEFAULT_JSON = (
    Path(__file__).resolve().parent.parent
    / "pyleecan"
    / "Data"
    / "Machine"
    / "IPMSM_B.json"
)


def load_machine(json_path: Path) -> dict:
    """Load machine description from JSON file.

    Parameters
    ----------
    json_path: Path
        Location of the machine description file

    Returns
    -------
    dict
        Parsed JSON content
    """

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Machine file not found: {json_path}. "
            "Use --json-path to specify a valid machine description"
        )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Initial values
def get_initial_values(data):
    return data["rotor"]["Rext"], data["stator"]["Rext"]


ROTOR_INITIAL = STATOR_INITIAL = None

# Bounds for design variables (meters)
ROTOR_BOUNDS = (0.05, 0.12)
STATOR_BOUNDS = (0.10, 0.16)
AIRGAP_MIN = 0.001  # minimal mechanical airgap

# Objective: maximise a dummy torque formula using rotor and stator radii


def evaluate(individual):
    rotor_r, stator_r = individual
    if rotor_r >= stator_r - AIRGAP_MIN:
        return (-1e6,)
    # Simplistic torque like function (area times airgap)
    torque = rotor_r**2 * (stator_r - rotor_r)
    return (torque,)


# Set up genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_rotor", random.uniform, *ROTOR_BOUNDS)
toolbox.register("attr_stator", random.uniform, *STATOR_BOUNDS)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attr_rotor, toolbox.attr_stator),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.4)


def bounded_gauss_mutation(individual, mu, sigma, indpb):
    for i, (val, (low, up)) in enumerate(
        zip(individual, [ROTOR_BOUNDS, STATOR_BOUNDS])
    ):
        if random.random() < indpb:
            val += random.gauss(mu, sigma)
            val = min(max(val, low), up)
            individual[i] = val
    return (individual,)


toolbox.register("mutate", bounded_gauss_mutation, mu=0, sigma=0.005, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main(n_gen=20, pop_size=20):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda fits: sum(fits) / len(fits))
    stats.register("max", max)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]
    print(f"Initial rotor Rext: {ROTOR_INITIAL} m")
    print(f"Initial stator Rext: {STATOR_INITIAL} m")
    print(f"Optimized rotor Rext: {best[0]:.5f} m")
    print(f"Optimized stator Rext: {best[1]:.5f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMSM genetic optimization example")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to machine JSON description",
    )
    args = parser.parse_args()

    mach_data = load_machine(args.json_path)
    ROTOR_INITIAL, STATOR_INITIAL = get_initial_values(mach_data)

    main()
