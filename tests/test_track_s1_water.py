import pytest
import numpy as np
from context import track_s1_water as tw


@pytest.fixture
def sample_result():
    a = np.array([
        [4081, 4085, 4093, 4097, 4109],
        [],
        [],
        [],
        [],
        [],
        [4081, 4085, 4093, 4097, 4109],
        [],
        [],
        []
    ])

    return a


def test_voronoi():
    assigned_sites = tw.assign_water_sites_voronoi(trj_file,
                                                   ref_file,
                                                   w_loc_ids,
                                                   w_names,
                                                   ref_file_water=None,
                                                   verbose=False)
