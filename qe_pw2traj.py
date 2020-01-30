'''
Copyright (C) 2013 SUNCAT This file is distributed under the terms of the GNU
General Public License. See the file `COPYING' in the root directory of the
present distribution, or http://www.gnu.org/copyleft/gpl.txt .

Convert QE output to ASE form
'''

import os
import numpy as np
import ase.io
from ase import Atoms
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from .qe_units import hartree, rydberg, bohr


def view_output(output_name):
    '''
    This function will try to read a QE output file and then use
    `ase.visualize.view` to look at the trajectory.

    Arg:
        output_name     A string indicating the path of the output file you
                        want to visualize
    '''
    images = read_positions_qe(output_name)
    view(images)


def write_traj(qe_log_name=None, output_traj_name='all.traj'):
    '''
    This is the main function in this submodule. It parses a Quantum Espresso
    log file and then writes all of the images into a single trajectory file.

    Args:
        qe_log_name         A string indicating the path of the Quantum Espress
                            log file. If `None` then looks for a
                            `firework-*.log` file.
        output_traj_name    A string indicating the desired name/path of the
                            output trajectory file
    Returns:
        images  A list of `ase.Atoms` objects of the trajectory
    '''
    # When defaulting, look for a 'fireworks-*.log' file
    if qe_log_name is None:
        qe_log_name = _find_qe_output_name()

    check_for_completion(qe_log_name)
    images = read_positions_qe(qe_log_name)
    ase.io.write(output_traj_name, images)
    return images


def _find_qe_output_name():
    '''
    This function will assume that you're using FireWorks with output file
    names of `fireworks-*.out`, and then find the exact name of the output file
    for you.

    Returns:
        file_   A string indicating the file name we found.
    '''
    for file_ in os.listdir():
        file_name = file_.split('.')[0]
        file_extension = file_.split('.')[-1]
        if 'fireworks' in file_name and file_extension == 'out':
            return file_


def check_for_completion(qe_log_name):
    '''
    Reads an output file and verifies whether or not it actually finished.

    Arg:
        qe_log_name     A string indicating the path of the Quantum Espresso
                        log file. If `None` then looks for a `firework-*.log`
                        file.
    '''
    failed_convergence_flag = os.popen('grep \'convergence NOT achieved\' %s' % qe_log_name).readline()
    assert not failed_convergence_flag, ('Calculation did not converge; aborting the writing of traj file')

    walltime_flag = os.popen('grep \'Maximum CPU time exceeded\' %s' % qe_log_name).readline()
    assert not walltime_flag, ('Calculation hit the wall time; aborting the writing of traj file')


def read_positions_qe(qe_output_name):
    """
    Input a QE logfile name and it extracts the postions and cells to build a
    trajectory file.
    May only work for pw?
    """
    # initialize arrays and atoms object
    steps = extract_coordinates(qe_output_name)
    energies = get_total_energies(qe_output_name)
    nsteps = max(steps.keys()) + 1
    forces = get_forces(qe_output_name)

    images = []
    for istep in range(nsteps):
        symbols = [''.join(i for i in s if not i.isdigit())
                   for s in [i[0] for i in steps[istep]['positions']]]
        positions = [i[1] for i in steps[istep]['positions']]
        mask = [len(i) != 3 for i in positions]
        posclean = [i[:3] for i in positions]
        cell = steps[istep]['cell']
        atoms = Atoms(symbols, posclean, cell=cell, pbc=(1, 1, 1))
        atoms.set_constraint(FixAtoms(mask=mask))
        calc = SinglePointCalculator(
            atoms=atoms,
            energy=energies[istep],
            forces=forces[istep],
            stress=None,
            magmoms=None)
        atoms.set_calculator(calc)
        images.append(atoms)

    return images


def extract_coordinates(qe_output_name):
    """
    Input a QE outfile and it extracts the atomic coordinates and the cell
    vectors for each ionic step.
    Returns the formatted dictionary for each step.

    Scales all units into Angstrom from bohr;
    ***  be careful with implementation since   ***
    ***  pwscf does not automatically write all ***
    ***  coordinates in bohr, while CP does     ***
    """
    # establish QE code (pw.x/cp.x) and version
    qeprogram, qeversion = [i.split() for i in os.popen(
        'grep Program %s' % (qe_output_name)).readlines()][0][1:3]
    if qeprogram == 'PWSCF':
        unitscale = 1.      # pw
        lvscale = float(
            os.popen(
                'grep \' lattice parameter (alat)\' %s' %
                (qe_output_name)).readline().split()[4]) * bohr
        poscmd = 'sed -e \'/./{H;$!d;}\' -e \'x;/ATOMIC_POS/!d;\' %s | sed -e \'/^ *$/d\' ' % (
            qe_output_name)
        cellcmd = 'sed -e \'/./{H;$!d;}\' -e \'x;/CELL_PARAM/!d;\' %s | sed -e \'/^ *$/d\' ' % (
            qe_output_name)
        # sed -e '/./{H;$!d;}' -e 'x;/CELL_PARAM/!d;' %s | sed -e '/^ *$/d' '
        # %(qe_output_name)
        celltranspose = 0

    elif qeprogram == 'CP':
        unitscale = bohr    # cp
        lvscale = 1.0     # updated later for non vc-relax pwscf
        poscmd = 'sed -n \'/ATOMIC_POS/,/ATOMIC_VEL/p\' %s | sed -e \'/VELOCITIES/d\' -e \'/^ *$/d\' ' % (
            qe_output_name)
        cellcmd = 'sed -n \'/   CELL_PARAM/,/System/p\' %s | sed -e \'/System/d\' -e \'/^ *$/d\' ' % (
            qe_output_name)
        # sed -n '/   CELL_PARAM/,/System/p ' %s | sed -e '/System/d' -e '/^
        # *$/d'
        celltranspose = 1

    print(
        "%s calculation: scaling lattice coordinates by %f" %
        (qeprogram, unitscale))

    # extract atomic positions and lattice parameters ( *** add a check for atomic coordinates ***)
    # PW writes lattice vectors as
    #   CELL_PARAMETERS (alat = lattice constant in bohr)
    #     avec
    #     bvec
    #     cvec
    # while CP writes them as
    #   CELL_PARAMETERS
    #     avec bvec cvec
    # i.e. they are transposed and always default to bohr

    poscoord = os.popen(poscmd)
    posraw = [i for i in poscoord.readlines()]
    pos = [i.split() for i in posraw if 'final' not in i]
    poslineno = [i for i, line in enumerate(pos) for word in line if 'POSITIONS' in word]

    # establish absolute or relative coordinates
    if 'crystal' in pos[0][-1]:
        pscale = lvscale
        print("crystal coordinates")
    elif 'angstrom' or 'bohr' in pos[0][-1]:
        pscale = 1.
        print("angstrom/bohr coordinates")

    # extract cell lattice vectors
    cellcoord = os.popen(cellcmd)
    cellraw = [i for i in cellcoord.readlines()]
    # works for CP MD or variable-cell PWSCF/CP (e.g., bulk relaxations)
    if cellraw:
        celllineno = [i for i, val in enumerate(cellraw) if 'CELL' in val]
        scell = [i.split() for i in cellraw]
    # PWSCF non variable-cell (e.g., everything else)
    else:
        cellcoord = os.popen('sed -n \'/a(1)/,/a(3)/p \' %s' % (qe_output_name))
        cellraw = [i for i in cellcoord.readlines()]
        cellraw = cellraw[-3:]
        nsw = len(poslineno)
        celllineno = [i * 4 for i in range(nsw)]
        # similar formatting to others but without CELL
        scell = [['CELL_PARAMETERS']] + [i.split()[3:6] for i in cellraw]
        scell = nsw * scell

    # scale vectors and units appropriately
    latscale = unitscale * pscale
    # Usually works with variable cell sizes
    try:
        if 'angstrom' in scell[0][1]:
            cellscale = 1.
        else:
            cellscale = unitscale * lvscale
    # Usually works with static cell sizes
    except IndexError:
        cellscale = unitscale * lvscale

    # split positions and cell into each step
    nsteps = len(poslineno) - 1

    steps = {}
    if nsteps == 0:
        steps[0] = {}
        # assumes that the ATOMIC_POSITIONS is still included, (why the +1 is
        # there)
        pos_unformatted = pos[1:]
        # assumes that the ATOMIC_POSITIONS is still included, (why the +1 is
        # there)
        cell_unformatted = scell[1:]
        steps[0]['positions'] = format_positions(pos_unformatted, latscale)
        steps[0]['cell'] = format_cell(
            cell_unformatted, cellscale, celltranspose)
    else:
        for i in range(nsteps):
            steps[i] = {}
            # assumes that the ATOMIC_POSITIONS is still included, (why the +1
            # is there)
            pos_unformatted = pos[poslineno[i]:poslineno[i + 1]]
            # assumes that the ATOMIC_POSITIONS is still included, (why the +1
            # is there)
            cell_unformatted = scell[celllineno[i] + 1:celllineno[i + 1]]
            steps[i]['positions'] = format_positions(pos_unformatted, latscale)
            steps[i]['cell'] = format_cell(
                cell_unformatted, cellscale, celltranspose)

        # add last step
        steps[i + 1] = {}
        # assumes that the ATOMIC_POSITIONS is still included, (why the +1 is
        # there)
        pos_unformatted = pos[poslineno[i + 1] + 1:]
        # assumes that the ATOMIC_POSITIONS is still included, (why the +1 is
        # there)
        cell_unformatted = scell[celllineno[i + 1] + 1:]
        steps[i + 1]['positions'] = format_positions(pos_unformatted, latscale)
        steps[i +
              1]['cell'] = format_cell(cell_unformatted, cellscale, celltranspose)
    return steps


def format_cell(cell_unformatted, cellscale=1.0, celltranspose=0):
    """ Enter in the unformatted lattice vectors as extracted from extract_coordinates
    Assumes formatting of [ ['a1x','a1y','a1z'], ['a2x','a2y','a2z'], ['a3x','a3y','a3z'] ]
    Returns np.array of cell.

    Can enter a scaling parameter to scale all returned lattice vectors.
    """
    cell_clean = cellscale * \
        np.array([[float(j) for j in val] for k, val in enumerate(cell_unformatted)])
    if celltranspose:
        cell_clean = cell_clean.transpose()
    return cell_clean


def format_positions(positions_unformatted, latscale=1.0):
    """ Enter in the unformatted positions as extracted from extract_coordinates.
    Assumes formatting of [ [ 'element', 'x', 'y', 'z'] ...]
    Returns [ [ 'element', np.array([x,y,z]) ] ... ]

    Can enter a scaling parameter to scale all returned positions.
    """
    positions = []
    for atomic_info in positions_unformatted:
        element = atomic_info[0]
        coords = atomic_info[1:4]
        try:
            coords = np.array([float(coords) for coords in coords])
        except ValueError:
            continue
        position = [element, coords]
        positions.append(position)
    return positions


def get_total_energies(filename):
    """ Extract total energies in cp (Ha) or pw (Ry) and convert to eV.
    """
    cpraw = os.popen('grep \'total energy =\' %s ' % (filename))
    cpe = [i.split() for i in cpraw.readlines()]
    if cpe:
        print("Identified a cp.x output file")
        energies = np.array([float(i[-3]) * hartree for i in cpe])

    pwraw = os.popen('grep \'! \' %s ' % (filename))
    pwe = [i.split() for i in pwraw.readlines()]
    # purge out lines about whether symmetry disabled
    pwe = [line for line in pwe if 'total' in line and 'energy' in line]
    if pwe:
        print("Identified a pw.x output file")
        energies = np.array([float(i[-2]) * rydberg for i in pwe])
    return energies


def get_forces(filename):
    '''
    Get the final atomic forces out of the output file.

    Arg:
        filename    A string indicating the name of the output file
    Returns:
        forces  A `SxNx3` numpy array, where S is the number of steps, N is the
                number of atoms, and each element corresponds to the force on
                atom `n` at step `s`.
    '''
    # Initialize
    n_images = get_number_of_images(filename)
    n_atoms = get_number_of_atoms(filename)
    forces = np.empty((n_images, n_atoms, 3))

    # Read the forces line-by-line and then store them in the output array
    all_forces = os.popen('grep \'  force =\' %s' % filename)
    for step_num in range(n_images):
        for atom_num in range(n_atoms):
            grepped_line = all_forces.readline()
            atomic_forces = grepped_line.split()[-3:]
            atomic_forces = [float(force) for force in atomic_forces]
            forces[step_num, atom_num] = atomic_forces
    return forces


def get_number_of_images(filename):
    '''
    Get the number of steps that the relaxation took, assuming that it uses
    BFGS.

    Arg:
        filename    A string indicating the name of the output file
    Returns:
        n_images    An integer for the total number of images, including the
                    initial
    '''
    qeprogram, qeversion = [i.split() for i in os.popen('grep Program %s' % (filename)).readlines()][0][1:3]
    if qeprogram == 'PWSCF':
        poscmd = 'sed -e \'/./{H;$!d;}\' -e \'x;/ATOMIC_POS/!d;\' %s | sed -e \'/^ *$/d\' ' % (filename)
    elif qeprogram == 'CP':
        poscmd = 'sed -n \'/ATOMIC_POS/,/ATOMIC_VEL/p\' %s | sed -e \'/VELOCITIES/d\' -e \'/^ *$/d\' ' % (filename)
    poscoord = os.popen(poscmd)
    posraw = [i for i in poscoord.readlines()]
    poslineno = [i for i, val in enumerate(posraw) if 'POSITIONS' in val]
    nsteps = len(poslineno)
    return nsteps


def get_number_of_atoms(filename):
    '''
    Get the number of atoms reported in the output file.

    Arg:
        filename    A string indicating the name of the output file
    Returns:
        n_steps     An integer for the total number of atoms
    '''
    n_atoms_line = os.popen('grep \'number of atoms\' %s' % filename).readline()
    n_atoms = int(n_atoms_line.split()[-1])
    return n_atoms
