#!/usr/bin/env python3

import unittest
import warnings

import numpy as np
from scipy.stats import rankdata

from lir.algorithms.percentile_rank import PercentileRankTransformer


warnings.simplefilter('error')


class TestPercentileRankTransformer(unittest.TestCase):
    def test_fit_transform(self):
        """When X itself is transformed, it should return its own ranks"""
        X = np.array([[0.1, 0.4, 0.5], [0.2, 0.5, 0.55], [0.15, 0.51, 0.55], [0.18, 0.45, 0.56]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(X)
        self.assertSequenceEqual(
            ranks.tolist(),
            (rankdata(X, method='max', axis=0) / len(X)).tolist(),
            'Ranking X and PercentileRankTransformer.transform(X) should give the same results',
        )

    def test_extrapolation(self):
        """Values smaller than the lowest value should map to 0,
        values larger than the highest value should map to 1"""
        X = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.4], [0.3, 0.2, 0.5]])
        Z = np.array([[0.0, 0.1, 0.2], [1.0, 1.0, 1.0]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertSequenceEqual(
            ranks.tolist(),
            [[0, 0, 0], [1, 1, 1]],
            'Elements smaller than the lowest value shouldmap to 0, larger than the highest value to 1',
        )

    def test_interpolation(self):
        """Values inbetween values of X result in interpolated ranking,
        with linear interpolation."""
        X = np.array([[0, 0, 0], [1, 1, 1]])
        Z = np.array([[0.1, 0.3, 0.5]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        # Ranks are interpolated between 0.5 (rank of 0) and 1 (rank of 1).
        # We expect a linear interpolation.
        expected_ranks = 0.5 + np.array([[0.1, 0.3, 0.5]]) * 0.5
        self.assertSequenceEqual(ranks.tolist(), expected_ranks.tolist(), 'Interpolation failed.')

    def test_ties(self):
        """Ties are given the maximum value (the maximum of the ranks that would
        have been assigned to all the tied values is assigned to each value)."""
        X = np.array([[0.1], [0.1], [0.1], [0.1], [0.3]])
        Z = np.array([[0.1]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertEqual(ranks, 0.8, 'Ties should be given the maximum value')

    def test_constant_feature(self):
        """If a feature is a constant value c, the rank should be 0 for x < c
        and 1 for x >= c."""
        X = np.array([[1], [1], [1], [1]])
        Z = np.array([[0], [1], [2]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertEqual(
            ranks.tolist(),
            [[0], [1], [1]],
            'If a feature is a constant value, interpolation should still work',
        )


if __name__ == '__main__':
    unittest.main()
