import numpy as np
import mdtraj as md
import os


def assign_water_sites_voronoi(trj_file, ref_file, w_loc_ids, w_names, ref_file_water=None, verbose=False):
    '''
    Assign one water oxygen atom to every specified water site for each frame based on Voronoi tesselation.

    Parameters
    ----------
    trj_file : str
        Path to trajectory file, e.g. .xtc.
    ref_file : str
        Path to trajectory topology file, e.g. .pdb.
    w_loc_ids : list
        Contains ids within ref_file_water of reference water oxygens. Used to define sites. Must have
        same length as w_names.
    w_names : list
        Contains names of water sites. Must have same length as w_loc_ids.
    ref_file_water : str or None
        Path to structure file (e.g. .pdb) that contains the locations of the reference water sites. If None,
        this will be equal to ref_file.
    verbose : bool
        Turn on/off verbose mode.

    Returns
    -------
    assigned_sites : np.ndarray
        Shape of (trj.n_frames, n_wat). Has the corresponding water oxygen atom id for each site and frame.
    '''
    # make sure w_loc_ids and w_names have same length
    if not len(w_loc_ids) == len(w_names):
        raise ValueError(
            f'w_loc_ids ({len(w_loc_ids)}) and w_names ({len(w_names)}) do not have same length.')

    # set trajectory topology pdb file for reference if not provided
    if not ref_file_water:
        ref_file_water = ref_file

    # load trj and reference trj
    t = md.load(trj_file, top=ref_file)
    t_ref = md.load(ref_file_water)

    n_wat = len(w_names)

    if verbose:
        print(f'Working directory:\t{os.getcwd()}')
        print(f'Trajectory:\t{trj_file}')
        print(f'Topology:\t{ref_file}')
        print(f'Trajectory has {t.n_frames} frames and {t.n_atoms} atoms')
        if ref_file_water:
            print(f'Reference file for water sites:\t{ref_file_water}')
        else:
            print('No reference file for water sited given, topology file will be used.')
        print('Atoms for water sites in reference:')
        for i, id in enumerate(w_loc_ids):
            print(id, t_ref.top.atom(id), w_names[i])

    # add virtual oxygen atoms for locations of reference water oxygens
    t.top.add_chain()
    # count chains
    n_chains = 0
    for c in t.top.chains:
        n_chains += 1
    # add one residue for all virtual oxygens to last chain
    t.top.add_residue('VS', t.top.chain(n_chains - 1))
    # add one virtual oxygen to last residue for each water site
    for i, name in enumerate(w_names):
        t.top.add_atom(f'VS-{name}', md.element.oxygen, t.top.residue(t.n_residues - 1))

    # store atom ids for virtual oxygens in trajectory
    w_ids = []
    for i_a, a in enumerate(t.top.residue(t.n_residues - 1).atoms):
        w_ids.append(a.index)
        if not a.name == f'VS-{w_names[i_a]}':
            raise ValueError(f'Atom {a.name} is not correctly indexed.')

    if verbose:
        print('Created virtual atoms in trajectory:')
        for idx in w_ids:
            print(idx, t.top.atom(idx))

    # add coordinates of virtual oxygens to trajectory
    coord_init = np.zeros((t.n_frames, n_wat, 3))
    xyz_new = np.concatenate((t.xyz, coord_init), axis=1)
    t.xyz = xyz_new

    for w_id in w_ids:
        # if not np.all(t.xyz[:, w_id] == np.zeros(len(t.xyz), 3)):
        # raise ValueError(f'Virtual oxygen {w_id} not correctly indexed.')

        if w_id == 0:
            raise ValueError('Virtual oxygens canot have index 0.')

    # get coordinates from reference file
    w_coord = np.zeros((n_wat, 3))
    for i, w_loc_id in enumerate(w_loc_ids):
        w_loc = t_ref.xyz[0, w_loc_id]
        w_coord[i] = w_loc

    # set coordinates in trajectory
    for i, w_id in enumerate(w_ids):
        for frame in t.xyz:
            frame[w_id] = w_coord[i]

    # water oxygens within a 2 Angstrom radius (neighbors) of virtual sites
    neighs = []
    water_O_ids = t.top.select('is_water and element == O')
    for i_wat in range(n_wat):
        neigh = md.compute_neighbors(t,
                                     0.2,
                                     query_indices=[w_ids[i_wat]],
                                     haystack_indices=water_O_ids,
                                     periodic=False)

        neighs.append(neigh)
    neighs = np.array(neighs)

    if verbose:
        # print average neighbor numbers for each virtual site
        print('Number of neighbors:')
        print('NAME\tMEAN\tSTD\tMIN\tMAX')
        for i, w_name in enumerate(w_names):
            lens = np.array([len(x) for x in neighs[i]])
            print(f'{w_name}:\t{np.mean(lens):.1f}\t{np.std(lens):.1f}\t{min(lens)}\t{max(lens)}')

    # initialize the assigned sites array
    assigned_sites = np.zeros((t.n_frames, n_wat), dtype=int)

    for i_frame in range(t.n_frames):
        neighs_frame = [x[i_frame] for x in neighs]
        assigned_sites_frame = np.zeros(n_wat, dtype=int)

        # get unique water ids from all neighbor lists
        unique_ids = np.unique(np.concatenate(neighs_frame))

        # distances of unique waters to every virtual oxygen
        dists = {}

        # iterate over every unique water id
        for uid in unique_ids:
            occures_in = np.zeros(n_wat, dtype=bool)
            dists[uid] = np.full(n_wat, 1000000, dtype=float)

            for i_wat in range(n_wat):
                if np.any(np.in1d(neighs_frame[i_wat], uid)):
                    occures_in[i_wat] = True
                    dists[uid][i_wat] = _dist(t.xyz, i_frame, uid, w_ids[i_wat])

            # keep id only in the neighbor list with the lowest distance to virtual oxygen (voronoi tesselation)
            i_wat_min = np.where(dists[uid] == np.amin(dists[uid]))[0][0]
            for i_wat in range(n_wat):
                if uid in neighs_frame[i_wat]:
                    if i_wat != i_wat_min:
                        neighs_frame[i_wat][neighs_frame[i_wat] == uid] = 0

        # iterate over neighbour lists
        for i_wat in range(n_wat):
            # remove ids in neighbour lists but the closest to virtual oxygen
            if np.count_nonzero(neighs_frame[i_wat]) > 1:

                dists_neli = np.full(len(neighs_frame[i_wat]), 100000)
                for i, idx in enumerate(neighs_frame[i_wat]):
                    dists_neli[i] = _dist(t.xyz, i_frame, idx, w_ids[i_wat])

                i_min = np.where(dists_neli == np.amin(dists_neli))[0][0]
                for i, idx in enumerate(neighs_frame[i_wat]):
                    if i != i_min:
                        neighs_frame[i_wat][neighs_frame[i_wat] == idx] = 0

        for i_wat in range(n_wat):
            # assign nonzero atom idx to sites
            if len(np.nonzero(neighs_frame[i_wat])[0]) == 0:
                assigned_sites_frame[i_wat] = 0
            else:
                assigned_sites_frame[i_wat] = neighs_frame[i_wat][neighs_frame[i_wat] != 0][0]

        assigned_sites[i_frame] = assigned_sites_frame

    return assigned_sites


def reduce_fluctuation(assigned_sites, t_l):
    '''
    Reduce fluctuation of water sites by counting the most frequent water of timeframe t_l to be present for the whole timeframe.

    Parameters
    ----------
    assigned_sites : np.ndarray
        Assigned_sites array.
    t_l : int
        Timestep.

    Returns
    -------
    assigned_sites_red : np.ndarray
        Assigned_sites array with reduced fluctuations.
    '''
    if not assigned_sites.ndim == 2:
        raise ValueError('assigned_sites must have 2 dimensions.')

    assigned_sites_red = np.zeros(assigned_sites.shape, dtype=int)
    n_frames = assigned_sites.shape[0]
    n_wat = assigned_sites.shape[1]

    for i_wat in range(n_wat):
        assigned_wat = np.zeros(n_frames, dtype=int)

        for i_frame in range(0, n_frames, t_l):
            # find most frequent value in timeframe
            slice = assigned_sites[:, i_wat][i_frame:i_frame + t_l]
            idx_max = np.bincount(slice).argmax()
            assigned_wat[i_frame:i_frame + t_l] = np.full(t_l, idx_max)

        assigned_sites_red[:, i_wat] = assigned_wat

    return assigned_sites_red


def count_transitions(assigned_sites, count_zero=True, return_zero=False, verbose=False):
    '''
    Count transitions in occupation of water sites.

    Parameters
    ----------
    assigned_sites : np.ndarray
        Assigned_sites array.
    count_zero : bool
        Count transitions from and to empty site.
    return_zero : bool
        Return count of transitions from and to empty site.
    verbose : bool
        Verbose mode.
    '''
    if not assigned_sites.ndim == 2:
        raise ValueError('assigned_sites must have 2 dimensions.')

    n_frames = assigned_sites.shape[0]
    n_wat = assigned_sites.shape[1]
    counts = np.zeros(n_wat, dtype=int)
    if return_zero:
        counts_zero = np.zeros(n_wat, dtype=int)

    for i_wat in range(n_wat):
        for i_frame in range(1, n_frames):
            idx_curr = assigned_sites[i_frame, i_wat]
            idx_prev = assigned_sites[i_frame - 1, i_wat]

            if idx_prev != idx_curr:
                if count_zero:
                    counts[i_wat] += 1
                else:
                    if idx_prev != 0 and idx_curr != 0:
                        counts[i_wat] += 1

                if return_zero:
                    if idx_prev != 0 and idx_curr != 0:
                        counts_zero[i_wat] += 1

    if not return_zero:
        return counts
    else:
        return counts, counts_zero


def residence_time(assigned_sites, timestep=1):
    '''
    Estimate residence time of waters at the sites.

    Parameters
    ----------
    assigned_sites : np.ndarray
        Assigned_sites array.
    timestep : int
        Trajectory timestep in picoseconds.

    Returns
    -------
    residence_times : np.ndarray
        Residence times of waters at sites in picoseconds.
    '''
    pass


def plot_transition_counts(ax, counts, w_names):
    '''
    Plot tranistion counts.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib ax.
    counts : dict
        Transition counts for each trj.
    w_names : list of str
        Water letter codes.
    '''
    ind = np.arange(len(w_names))
    width = 0.1

    ind_clip = len(counts) / 2 - 0.5
    ind_bar = np.arange(-ind_clip, ind_clip + 1, 1)

    colors = ['k', 'dimgrey', 'grey', 'silver']

    for i_key, key in enumerate(counts):
        means = np.mean(counts[key], axis=0)
        errs = np.std(counts[key], axis=0)

        ax.bar(ind + width * ind_bar[i_key],
               means,
               width,
               yerr=errs,
               label=key,
               color=colors[i_key % 4],
               edgecolor='k')

    ax.set_xticks(ind)
    ax.set(xticklabels=w_names)
    ax.set_ylabel('Number of Water Molecules at Site')
    ax.set_xlabel('S1 Water Site')
    ax.legend()


def _dist(xyz, frame, idx1, idx2):
    '''
    Calculate distance between two atoms with given indices with trajectory coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Trajectory coordinates.
    frame : int
        Frame number.
    idx1, idx2 : int
        Atom indices.

    Returns
    -------
    dist : float
        Distance between the two atoms in nanometers.
    '''
    coord_1 = xyz[frame][idx1]
    coord_2 = xyz[frame][idx2]
    return np.linalg.norm(coord_1 - coord_2)
