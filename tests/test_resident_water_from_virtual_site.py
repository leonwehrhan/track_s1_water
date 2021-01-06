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

@pytest.fixture
def trajectory_files():
    trj_file = 'data/E3G_complex_short.xtc'
    ref_file = 'data/E3G_complex_short.pdb'

    return trj_file, ref_file

def test_ResidentWater():

    a = tw.ResidentWater()
    a.start = 2
    a.stop = 10
    assert a.residence_frames() == 8

def test_compute_neighbors_p1(trajectory):
    t = trajectory

    water_O_ids = t.top.select('is_water and element == O')
    neighbor_list = md.compute_neighbors(t, 0.8, query_indices=[3415], haystack_indices=water_O_ids)

    assert (neighbor_list[0] == np.array([4201, 4273, 4397, 4501, 4625, 4629, 5113])).all()

def test_get_residence_waters(trajectory):
    t = trajectory

    water_O_ids = t.top.select('is_water and element == O')
    neighbor_list = md.compute_neighbors(t, 0.8, query_indices=[3415], haystack_indices=water_O_ids)

    resident_waters_short = tw.get_resident_waters(neighbor_list, n_strikes=1)

    for id in [4201, 4273, 4397, 4501, 4625, 4629, 5113]:
        assert any(rw.id == id for rw in resident_waters_short if rw.start == 0)

    assert any([rw.residence_frames() > 0 for rw in resident_waters_short])

    counter_4201 = 0
    for rw in resident_waters_short:
        if rw.id == 4201:
            counter_4201 += 1
            assert rw.start == 0
            assert rw.stop == 9
    assert counter_4201 == 1

    counter_4501 = 0
    for rw in resident_waters_short:
        if rw.id == 4501:
            counter_4501 += 1
            assert rw.start == 0 or rw.start == 3
            assert rw.stop == 2 or rw.stop == 9
    assert counter_4501 == 2

def test_virtual_site(trajectory):
    t = trajectory

    n_atoms = t.n_atoms
    assert n_atoms == 47301

    t.top.add_chain()
    t.top.add_residue('VIR', t.top.chain(3))
    t.top.add_atom('VIR-O', md.element.oxygen, t.top.residue(t.n_residues-1))

    zers = np.zeros((t.n_frames, 1, 3))
    xyz_new = np.concatenate((t.xyz, zers), axis=1)
    t.xyz = xyz_new

    assert t.xyz.shape[1] == t.n_atoms

    for frame in t.xyz:
        frame[t.n_atoms-1] = 0.5*(frame[3415]-frame[2473])+frame[2473]

    n_chains = 0
    for c in t.top.chains:
        n_chains += 1
    assert n_chains == 4

    assert t.n_residues == 11095
    assert t.n_atoms == 47302

    n_atoms_res = 0
    for a in t.top.residue(11094).atoms:
        n_atoms_res += 1
    assert n_atoms_res == 1

    for frame in t.xyz:
        assert round(np.linalg.norm(frame[3415]-frame[47301]), 3) == round(np.linalg.norm(frame[2473]-frame[47301]), 3)


