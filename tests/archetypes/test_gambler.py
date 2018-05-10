import axelrod as axl
import random
import unittest

from axelrod_dojo import GamblerParams
from axelrod.strategies.lookerup import Plays

C, D = axl.Action.C, axl.Action.D


class TestGamblerParams(unittest.TestCase):
    def test_init(self):
        plays = 2
        op_plays = 2
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertEqual(gambler_params.PlayerClass, axl.Gambler)
        self.assertEqual(gambler_params.plays, plays)
        self.assertEqual(gambler_params.op_plays, op_plays)
        self.assertEqual(gambler_params.op_start_plays, op_start_plays)

    def test_player(self):
        plays, op_plays, op_start_plays = 1, 1, 1
        pattern = [0 for _ in range(8)]
        gambler = GamblerParams(plays=plays, op_plays=op_plays,
                                op_start_plays=op_start_plays, pattern=pattern)

        player =  gambler.player()
        action_dict = {Plays(self_plays=(C,), op_plays=(C,), op_openings=(C,)): 0,
                       Plays(self_plays=(C,), op_plays=(C,), op_openings=(D,)): 0,
                       Plays(self_plays=(C,), op_plays=(D,), op_openings=(C,)): 0,
                       Plays(self_plays=(C,), op_plays=(D,), op_openings=(D,)): 0,
                       Plays(self_plays=(D,), op_plays=(C,), op_openings=(C,)): 0,
                       Plays(self_plays=(D,), op_plays=(C,), op_openings=(D,)): 0,
                       Plays(self_plays=(D,), op_plays=(D,), op_openings=(C,)): 0,
                       Plays(self_plays=(D,), op_plays=(D,), op_openings=(D,)): 0}
        
        self.assertEqual(player.lookup_dict, action_dict)

    def test_receive_vector(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertRaises(AttributeError, GamblerParams.__getattribute__,
                          *[gambler_params, 'vector'])

        vector = [random.random() for _ in range(6)]
        gambler_params.receive_vector(vector)

        self.assertEqual(gambler_params.pattern, vector)

    def test_vector_to_instance(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        vector = [random.random() for _ in range(8)]
        gambler_params.receive_vector(vector)
        instance = gambler_params.player()

        action_dict = {Plays(self_plays=(C,), op_plays=(C,), op_openings=(C,)): vector[0],
                       Plays(self_plays=(C,), op_plays=(C,), op_openings=(D,)): vector[1],
                       Plays(self_plays=(C,), op_plays=(D,), op_openings=(C,)): vector[2],
                       Plays(self_plays=(C,), op_plays=(D,), op_openings=(D,)): vector[3],
                       Plays(self_plays=(D,), op_plays=(C,), op_openings=(C,)): vector[4],
                       Plays(self_plays=(D,), op_plays=(C,), op_openings=(D,)): vector[5],
                       Plays(self_plays=(D,), op_plays=(D,), op_openings=(C,)): vector[6],
                       Plays(self_plays=(D,), op_plays=(D,), op_openings=(D,)): vector[7]}

        self.assertIsInstance(instance, axl.Gambler)
        self.assertEqual(instance.lookup_dict, action_dict)

    def test_create_vector_bounds(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)
        lb, ub = gambler_params.create_vector_bounds()

        self.assertIsInstance(lb, list)
        self.assertIsInstance(ub, list)
        self.assertEqual(len(lb), 8)
        self.assertEqual(len(ub), 8)

    def test_random_params(self):
        plays = 2
        op_plays = 2
        op_start_plays = 2

        table = GamblerParams.random_params(plays=plays, op_plays=op_plays,
                                            op_start_plays=op_start_plays)

        self.assertIsInstance(table, list)
        self.assertEqual(len(table), 64)

    def test_randomize(self):
        plays = 2
        op_plays = 2
        op_start_plays = 2

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertIsInstance(gambler_params.pattern, list)
        self.assertEqual(len(gambler_params.pattern), 64)

    def test_repr(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        pattern = [0.05, 0.07, 0.70, 0.03,
                   0.60, 0.70, 0.45, 0.10]

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays,
                                       pattern=pattern)

        self.assertEqual(gambler_params.__repr__(),
                         '1:1:1:0.05|0.07|0.7|0.03|0.6|0.7|0.45|0.1')

    def test_crossover(self):
        plays=1
        op_plays=1
        op_start_plays=1

        gambler1 = GamblerParams(plays=plays, op_plays=op_plays,
                                 op_start_plays=op_start_plays)

        gambler2 = GamblerParams(plays=plays, op_plays=op_plays,
                                 op_start_plays=op_start_plays)

        axl.seed(0)
        gambler = gambler1.crossover(gambler2)
        self.assertEqual(gambler.pattern,
                         gambler1.pattern[:3] + gambler2.pattern[3:])

        self.assertEqual(gambler.PlayerClass, gambler1.PlayerClass)
        self.assertEqual(gambler.plays, gambler.plays)
        self.assertEqual(gambler.op_plays, gambler.op_plays)
        self.assertEqual(gambler.op_start_plays, gambler.op_start_plays)

    def test_mutations(self):
        gambler_params = GamblerParams(plays=1, op_plays=1, op_start_plays=1,
                                       mutation_probability=0.3)
        pattern = [i for i in gambler_params.pattern]

        axl.seed(0)
        _ = gambler_params.mutate()

        self.assertNotEqual(pattern, gambler_params.pattern)
        self.assertEqual(pattern[:3], gambler_params.pattern[:3])
        self.assertEqual(pattern[4:], gambler_params.pattern[4:])

    def test_genes_in_bounds(self):
        gambler_params = GamblerParams(plays=1, op_plays=1, op_start_plays=1,
                                       mutation_probability=0.8)

        for _ in range(100):
            _ = gambler_params.mutate()

        self.assertTrue(all(gene <= 1 for gene in gambler_params.pattern))
        self.assertTrue(all(gene >= 0 for gene in gambler_params.pattern))