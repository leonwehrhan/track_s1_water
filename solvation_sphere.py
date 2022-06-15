import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import pickle
import os


def dist(coord1, coord2):
    '''Distance between two atoms defined by their coordinates.'''
    return np.abs(np.linalg.norm(coord1 - coord2))


def angle_(t, O, H, A, frame=0):
    '''O-H-A angle between water and acceptor in mdtraj trajectory. Atoms defined by index in topology.'''
    vec1 = t.xyz[frame][H] - t.xyz[frame][O]
    vec2 = t.xyz[frame][H] - t.xyz[frame][A]
    dot_product = np.dot(unit_vector(vec1), unit_vector(vec2))
    angle = np.degrees(np.arccos(dot_product))
    return angle


def unit_vector(v):
    '''Normalize vector.'''
    return v / np.linalg.norm(v)


def find_H(t, idx):
    '''Find hydrogens that belong to a water oxygen in a mdtraj topology. Works also for other residues than water.'''
    Hs = []
    a = t.top.atom(idx)
    r = a.residue
    for x in r.atoms:
        if x.element == md.element.hydrogen:
            Hs.append(x.index)
    return Hs

def solvation_dist_ang(t, rw_assigned, i_F):
    '''
    Get distance and angle timeseries of S1 waters to fluorine.

    Parameters
    ----------
    t : md.Trajectory.
        MD trajectory.
    rw_assigned : np.ndarray
        Assignment of S1 waters to oxygen atoms in t.
    i_F : list
        Indices of fluorine atoms.

    Returns
    -------
    dists : list of np.ndarray
        Distances to S1 waters for each fluorine seperately.
    angs : list of np.ndarray
        Angles to S1 waters for each fluorine seperately.
    '''
    dists = []
    angs = []

    for idx_F in i_F:
        dists_F = np.zeros((t.n_frames, 5))
        for i_wat in range(5):
            for i_frame in range(t.n_frames):
                idx_O = rw_assigned[i_frame][i_wat]
                d = dist(t.xyz[i_frame][idx_F], t.xyz[i_frame][idx_O])
                dists_F[i_frame][i_wat] = d
        dists.append(dists_F)

    for idx_F in i_F:
        angs_F = np.zeros((t.n_frames, 10))
        for i_wat in range(5):
            for i_frame in range(t.n_frames):
                idx_O = rw_assigned[i_frame][i_wat]
                Hs = find_H(t, idx_O)
                ang1 = angle_(t, idx_O, Hs[0], idx_F, frame=i_frame)
                ang2 = angle_(t, idx_O, Hs[1], idx_F, frame=i_frame)
                angs_F[i_frame][i_wat * 2] = ang1
                angs_F[i_frame][i_wat * 2 + 1] = ang2
        angs.append(angs_F)

    return dists, angs