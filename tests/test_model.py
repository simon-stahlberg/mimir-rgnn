import pymimir as mm
import pytest

from pathlib import Path
from pymimir_rgnn import RelationalGraphNeuralNetwork


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm, rand", [
    ('blocks', 'hmax', 2, 2, False, False, False),
    ('blocks', 'smax', 3, 3, False, False, True),
    ('blocks', 'mean', 4, 4, False, True, False),
    ('blocks', 'add', 5, 5, True, False, False),
    ('gripper', 'hmax', 2, 2, True, False, False),
    ('gripper', 'smax', 3, 3, True, False, True),
    ('gripper', 'mean', 4, 4, True, True, False),
    ('gripper', 'add', 5, 5, True, True, False),
])
def test_create_model(dom: str, agg: str, layers: int, size: int, gro: bool, norm: bool, rand: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    domain = mm.Domain(domain_path)
    model = RelationalGraphNeuralNetwork(domain, agg, layers, size, gro, norm, rand)
    assert model is not None


@pytest.mark.parametrize("dom, agg, layers, size, gro, norm, rand", [
    ('blocks', 'hmax', 2, 2, False, False, False),
    ('blocks', 'smax', 3, 3, False, False, True),
    ('blocks', 'mean', 4, 4, False, True, False),
    ('blocks', 'add', 5, 5, True, False, False),
    ('gripper', 'hmax', 2, 2, True, False, False),
    ('gripper', 'smax', 3, 3, True, False, True),
    ('gripper', 'mean', 4, 4, True, True, False),
    ('gripper', 'add', 5, 5, True, True, False),
])
def test_forward_model(dom: str, agg: str, layers: int, size: int, gro: bool, norm: bool, rand: bool):
    domain_path = DATA_DIR / dom / 'domain.pddl'
    problem_path = DATA_DIR / dom / 'problem.pddl'
    domain = mm.Domain(domain_path)
    problem = mm.Problem(domain, problem_path)
    model = RelationalGraphNeuralNetwork(domain, agg, layers, size, gro, norm, rand)
    initial_state = problem.get_initial_state()
    initial_actions = initial_state.generate_applicable_actions()
    original_goal = problem.get_goal_condition()
    input = [(initial_state, initial_actions, original_goal)]
    output = model.forward(input)
    assert isinstance(output, list)
    assert len(output) == 1
    assert len(output[0]) == len(initial_actions)
