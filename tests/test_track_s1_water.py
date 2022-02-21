import pytest
import numpy as np
from context import track_s1_water as tw


def test_voronoi():
    '''Test for assign_water_sites_voronoi.'''
    assigned_sites = tw.assign_water_sites_voronoi('data/4y0z_a.pdb',
                                                   'data/4y0z_t.pdb',
                                                   [4078, 4081, 4072, 4084, 4075],
                                                   ['A', 'B', 'C', 'D', 'E'],
                                                   ref_file_water=None,
                                                   verbose=False)

    result = np.array([[4078, 0, 0, 0, 0],
                       [0, 4081, 0, 0, 0],
                       [0, 0, 4072, 0, 0],
                       [0, 0, 0, 4084, 0],
                       [0, 0, 0, 0, 4075],
                       [4078, 4081, 4072, 4084, 4075],
                       [0, 0, 0, 0, 0]])
    assert (assigned_sites == result).all()


def test_reduce_fluctuation():
    '''Test for reduce_fluctuation and count_transitions.'''
    assigned_sites = tw.assign_water_sites_voronoi('data/4y0z_fluc.pdb',
                                                   'data/4y0z_t.pdb',
                                                   [4078, 4081, 4072, 4084, 4075],
                                                   ['A', 'B', 'C', 'D', 'E'],
                                                   ref_file_water=None,
                                                   verbose=False)

    assigned_rf = tw.reduce_fluctuation(assigned_sites, 10)
    count = tw.count_transitions(assigned_rf)

    assert (count == np.array([1, 0, 0, 0, 0])).all()
