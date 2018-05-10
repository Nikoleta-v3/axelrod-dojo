import axelrod as axl
import axelrod_dojo as dojo
import functools
import numpy as np
import unittest

from axelrod_dojo import GamblerParams, prepare_objective, FSMParams
from axelrod_dojo.algorithms.particle_swarm_optimization import PSO


class TestPSO(unittest.TestCase):
    def test_init_default(self):
        params = [1, 1, 1]
        objective = prepare_objective('score', 2, 0, 1, nmoran=False)

        pso = PSO(GamblerParams, params, objective=objective)

        self.assertIsInstance(pso.objective, functools.partial)
        self.assertEqual(len(pso.opponents_information), len(axl.short_run_time_strategies))
        self.assertEqual(pso.population, 1)
        self.assertEqual(pso.generations, 1)
        self.assertTrue(pso.debug)
        self.assertEqual(pso.phip, 0.8)
        self.assertEqual(pso.phig, 0.8)
        self.assertEqual(pso.omega, 0.8)
        self.assertEqual(pso.processes, 1)

    def test_init(self):
        params = [2, 1, 1]
        objective = prepare_objective('score', 2, 0, 1, nmoran=False)
        opponents = [axl.Defector(), axl.Cooperator()]
        population = 2
        generations = 10
        debug = False
        phip = 0.6
        phig = 0.6
        omega = 0.6

        pso = PSO(GamblerParams, params, objective=objective,
                  opponents=opponents, population=population,
                  generations=generations, debug=debug, phip=phip, phig=phig,
                  omega=omega)

        self.assertIsInstance(pso.objective, functools.partial)
        self.assertEqual(len(pso.opponents_information), len(opponents))
        self.assertEqual(pso.population, population)
        self.assertEqual(pso.generations, generations)
        self.assertFalse(pso.debug)
        self.assertEqual(pso.phip, phip)
        self.assertEqual(pso.phig, phig)
        self.assertEqual(pso.omega, omega)
        self.assertEqual(pso.processes, 1)

    def test_pso_with_gambler(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_plays = 1
        num_op_plays = 1
        num_op_start_plays = 1
        params_kwargs = {"plays": num_plays,
                         "op_plays": num_op_plays,
                         "op_start_plays": num_op_start_plays}
        population = 10
        generations = 100
        opponents = [axl.Cooperator() for _ in range(5)]

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        pso = PSO(GamblerParams, params_kwargs, objective=objective, debug=False,
                  opponents=opponents, population=population, generations=generations)

        axl.seed(0)
        opt_vector, opt_objective_value = pso.swarm()

        self.assertTrue(np.allclose(opt_vector, np.array([[0.72231352, 0.29956341,
                                                           0.82648405, 0.46293523,
                                                           0.7069418, 0.85391998,
                                                           0, 0.050593]])))
        self.assertEqual(abs(opt_objective_value), 3.624)

    def test_pso_with_fsm(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 4
        params_kwargs = {"num_states": num_states}
        population = 10
        generations = 100
        opponents = [axl.Defector() for _ in range(5)]

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        pso = PSO(FSMParams, params_kwargs, objective=objective, debug=False,
                  opponents=opponents, population=population, generations=generations)

        axl.seed(0)
        opt_vector, opt_objective_value = pso.swarm()

        self.assertTrue(np.allclose(opt_vector, np.array([0.0187898, 0.6176355,
                                                          0.61209572, 0.616934,
                                                          0.94374808, 0.6818203,
                                                          0.3595079, 0.43703195,
                                                          0.6976312, 0.06022547,
                                                          0.66676672, 0.67063787,
                                                          0.21038256, 0.1289263,
                                                          0.31542835, 0.36371077,
                                                          0.57019677])))
        self.assertEqual(abs(opt_objective_value), 1)
