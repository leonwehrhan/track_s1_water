import numpy as np
import mdtraj as md
import pytest

from context import track_s1_water as tw


@pytest.fixture
def trajectory():
    trj_file = 'data/E3G_complex_short.xtc'
    ref_file = 'data/E3G_complex_short.pdb'

    t = md.load(trj_file, top=ref_file)

    return t


def test_voronoi(trajectory):
    t = trajectory

    assigned_sites = tw.assign_water_sites_voronoi(
        trj_file, ref_file, w_loc_ids, w_names, ref_file_water=None, verbose=False)
