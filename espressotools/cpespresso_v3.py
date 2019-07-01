"""
Copyright (C) 2013 SUNCAT This file is distributed under the terms of the GNU
General Public License. See the file `COPYING' in the root directory of the
present distribution, or http://www.gnu.org/copyleft/gpl.txt .

This is a modified version for the SUNCAT ASE-QE interface tailored for CP MD.
Most modifications are only to the setups, but many of the functionalities for
running through ASE have been deleted.  This is designed for standalone input
generation and only requires ASE and possibly the other qe_* functions.

Still a work in progress: look for FIX to identify places needing work.
"""

import os
import string
import numpy as np
from compiler.ast import flatten
from ase.calculators.general import Calculator
from ase import constraints

from .qe_units import rydberg, rydberg_over_bohr
# from (qe_)constants
# hartree = 27.21138505
# rydberg = 0.5*hartree
# bohr    = 0.52917721092
# rydberg_over_bohr = rydberg / bohr


def formatfreqs(freqs):
    """ temporary function to format the Nose frequencies to match the correct atomic ordering
    not guaranteed that Nose freqs entered in a list will necessary match atom-ordering ASE decides on
    these freqs could be stored in a dictionary like the atomic masses so that they are bound by atom type and not
    an arbitrary array.

    Ordering of frequencies should match the ordering of atomic species defined in ATOMIC_SPECIES with pseudopotentials
    """
    nosefreqs = string.join([str(i) for i in freqs], ' ')
    return nosefreqs


def mapkey(key, keytype):
    """ Designed to generate a string for entry into a QE input file based on ASE imput
    """
    print("update mapkey")


def _populate_specprop_helper(checkflag, symbols, rawval=[0]):
    """
    Function for helping to set index-dependent properties in espresso;
     e.g. setting Hubbard U, Nose-Hoover freqs, or RISM LJ parameters that depend on indices
     that correspond to ordering of pseudopotentials
    :param checkflag: this is the dictionary of properties that is being set (e.g. d_U = {'Cu' : 0.5, ...} )
    :param symbols: these are the atoms.get_chemical_symbols()
    :return: the property list that will define the property in atom2species
    """
    default_val = rawval * len(symbols)

    if checkflag is not None:
        if isinstance(checkflag, dict):
            proplist = default_val
            for i, s in enumerate(symbols):
                if s in checkflag:
                    proplist[i] = checkflag[s]
        else:
            proplist = list(checkflag)
            if len(proplist) < len(symbols):
                proplist += list([0] * (len(symbols) - len(proplist)))
    else:
        proplist = default_val
    return proplist


def find_max_empty_space(atoms, edir=3):
    iedir = edir - 1
    lenz = np.sum(atoms.get_cell()[:, iedir])
    minz = min([i[iedir] for i in atoms.positions])
    maxz = max([i[iedir] for i in atoms.positions])
    if minz < 0:
        newmaxz = lenz + minz
        newminz = maxz
    else:
        newmaxz = lenz
        newminz = maxz
    emaxpos = (maxz + (newmaxz - newminz) / 2.) / lenz  # noqa: F841

    comz = atoms.get_center_of_mass()[iedir]
    emaxcom = (comz + lenz / 2.) / lenz
    # print "com vac center: %.3f " %(emaxcom)
    # print "manual vac center: %.3f" %(emaxpos)
    return emaxcom


def num2str(x):
    """ Add 'd0' to floating point number to avoid random trailing digits in Fortran input routines
    From (qe_)utils
    """
    s = str(x)
    if s.find('e') < 0:
        s += 'd0'
    return s


def bool2str(x):
    """ Convert python to fortran logical
    From (qe_)utils
    """
    if x:
        return '.true.'
    else:
        return '.false.'


class specobj:
    """  small species class containing the attributes of a species
    ** updated from (qe_)utils ***
    """

    def __init__(
            self,
            s='X',
            mass=0.,
            magmom=0.,
            U=0.,
            J=0.,
            U_alpha=0.,
            nosefreq=0.,
            psptype=None,
            LJ_type=None,
            LJ_epsilon=None,
            LJ_sigma=None,
            qstart=None):
        self.s = s
        self.mass = mass
        self.magmom = magmom
        self.U = U
        self.J = J
        self.U_alpha = U_alpha
        self.nosefreq = nosefreq
        self.psptype = psptype
        self.LJ_epsilon = LJ_epsilon
        self.LJ_sigma = LJ_sigma
        self.LJ_type = LJ_type
        self.qstart = qstart


def convert_constraints(atoms):
    """ Convert some of ase's constraints to pw.x constraints for pw.x internal relaxation
    Returns constraints which are simply expressed as setting force components as first list
    and other contraints that are implemented in espresso as second list
    From (qe_)utils
    """
    if atoms.constraints:
        n = len(atoms)
        if n == 0:
            return [], []
        forcefilter = []
        otherconstr = []
        for c in atoms.constraints:
            if isinstance(c, constraints.FixAtoms):
                if len(forcefilter) == 0:
                    forcefilter = np.ones((n, 3), np.int)
                forcefilter[c.index] = [0, 0, 0]
            elif isinstance(c, constraints.FixCartesian):
                if len(forcefilter) == 0:
                    forcefilter = np.ones((n, 3), np.int)
                forcefilter[c.a] = c.mask
            elif isinstance(c, constraints.FixBondLengths):
                for d in c.constraints:
                    otherconstr.append(
                        "'distance' %d %d" %
                        (d.indices[0] + 1, d.indices[1] + 1))
            elif isinstance(c, constraints.FixBondLength):
                otherconstr.append(
                    "'distance' %d %d" %
                    (c.indices[0] + 1, c.indices[1] + 1))
            elif isinstance(c, constraints.FixInternals):
                # we ignore the epsilon in FixInternals because there can only be one global
                # epsilon be defined in espresso for all constraints
                for d in c.constraints:
                    if isinstance(
                            d, constraints.FixInternals.FixBondLengthAlt):
                        otherconstr.append(
                            "'distance' %d %d %s" %
                            (d.indices[0] +
                             1,
                             d.indices[1] +
                                1,
                                num2str(
                                d.bond)))
                    elif isinstance(d, constraints.FixInternals.FixAngle):
                        otherconstr.append(
                            "'planar_angle' %d %d %d %s" %
                            (d.indices[0] +
                             1,
                             d.indices[1] +
                                1,
                                d.indices[2] +
                                1,
                                num2str(
                                np.arccos(
                                    d.angle) *
                                180. /
                                np.pi)))
                    elif isinstance(d, constraints.FixInternals.FixDihedral):
                        otherconstr.append(
                            "'torsional_angle' %d %d %d %d %s" %
                            (d.indices[0] +
                             1,
                             d.indices[1] +
                                1,
                                d.indices[2] +
                                1,
                                d.indices[3] +
                                1,
                                num2str(
                                np.arccos(
                                    d.angle) *
                                180. /
                                np.pi)))
                    else:
                        raise NotImplementedError(
                            'constraint ' + d.__name__ + ' from FixInternals not implemented\n'
                            'consider ase-based relaxation with this constraint instead')
            else:
                raise NotImplementedError(
                    'constraint ' + c.__name__ + ' not implemented\n'
                    'consider ase-based relaxation with this constraint instead')
        return forcefilter, otherconstr
    else:
        return [], []


convergence_keys = ['convergence',      # include for adding the convergence flags to input file
                    'elec_conv_thr',
                    # self.convergence['electronic energy']  # QE conv_thr /
                    # VASP EDIFF
                    'ekin_conv_thr',
                    # self.convergence['kinetic energy']     # QE ekin_conv_thr
                    # for CP-MD runs
                    'etot_conv_thr',
                    # self.convergence['total energy']       # QE etot_conv_thr
                    # / VASP EDIFFG
                    'fmax',
                    # self.convergence['fmax']               # QE forc_conv_thr
                    'diag',
                    'scf_maxsteps',
                    'mixing',
                    'pressure',         # target pressure (kBar)
                    'constraints_tol',
                    # self.convergence['constraints']  constr_tol
                    'rism1d_conv_thr',
                    'rism3d_conv_thr',
                    'rism1d_maxstep',
                    'rism3d_maxstep']

# PW/CP keys
kpoint_keys = ['kpts', 'kptshift']
control_keys = ['calculation',    # pw/cp str
                'dipfield',       # pw bool
                'disk_io',        # pw/cp str
                'dt',             # pw/cp float
                'ekin_conv_thr',  # cp float
                'etot_conv_thr',  # pw/cp float
                'forc_conv_thr',  # pw/cp float
                'gate',           # pw bool
                'gdir',           # pw int
                'iprint',         # pw/cp int
                'isave',          # cp int
                'lberry',         # pw bool
                'lelfield',       # pw bool
                'lfcpopt',        # pw bool
                'lkpoint_dir',    # pw bool
                'lorbm',          # pw bool
                'max_seconds',    # pw/cp float
                'memory',         # cp str
                'nberrycyc',      # pw int
                'ndr',            # cp int
                'ndw',            # cp int
                'nppstr',         # pw int
                'nstep',          # pw/cp int
                'outdir',         # pw/cp str
                'prefix',         # pw/cp str
                'pseudo_dir',     # pw/cp str
                'restart_mode',   # pw/cp str
                'saverho',        # cp bool
                'tabps',          # cp bool
                'tefield',        # pw/cp bool
                'title',          # pw/cp str
                'tprnfor',        # pw/cp bool
                'tstress',        # pw/cp bool
                'verbosity',      # pw/cp str
                'wf_collect',     # pw bool
                'wfcdir',         # pw str
                ]

system_keys = ['A',                                  # pw/cp
               'B',                                  # pw/cp
               'C',                                  # pw/cp
               'Hubbard_J(i,ityp)',                  # pw
               'Hubbard_J0',                         # pw
               'Hubbard_U',                          # pw/cp
               'Hubbard_alpha',                      # pw
               'Hubbard_beta',                       # pw
               'U_projection_type',                  # pw str
               'angle1',                             # pw
               'angle2',                             # pw
               'assume_isolated',                    # pw/cp str
               'block',                              # pw bool
               'block_1',                            # pw float
               'block_2',                            # pw float
               'block_height',                       # pw float
               'celldm',                             # pw/cp
               'constrained_magnetization',          # pw str
               'cosAB',                              # pw/cp
               'cosAC',                              # pw/cp
               'cosBC',                              # pw/cp
               'degauss',                            # pw/cp float
               'eamp',                               # pw float
               'ecfixed',                            # pw/cp float
               'ecutfock',                           # pw float
               'ecutrho',                            # pw/cp float
               'ecutvcut',                           # pw float
               'ecutwfc',                            # pw/cp float
               'edir',                               # pw int
               'emaxpos',                            # pw float
               'eopreg',                             # pw float
               'esm_bc',                             # pw str
               'esm_efield',                         # pw float
               'esm_nfit',                           # pw int
               'esm_w',                              # pw float
               'exx_fraction',                       # pw/cp float
               'exxdiv_treatment',                   # pw str
               # 'fcp_mu',                             # pw float
               'fixed_magnetization',                # pw
               'force_symmorphic',                   # pw bool
               'ibrav',                              # pw/cp int
               'input_dft',                          # pw/cp str
               'lambda',                             # pw float
               'lda_plus_u',                         # pw/cp bool
               'lda_plus_u_kind',                    # pw int
               'london',                             # pw bool
               'london_c6',                          # pw
               'london_rcut',                        # pw/cp float
               'london_rvdw',                        # pw
               'london_s6',                          # pw/cp float
               'lspinorb',                           # pw bool
               'nat',                                # pw/cp int
               'nbnd',                               # pw/cp int
               'no_t_rev',                           # pw bool
               'noinv',                              # pw bool
               'noncolin',                           # pw bool
               'nosym',                              # pw bool
               'nosym_evc',                          # pw bool
               'nqx1',                               # pw int
               'nqx2',                               # pw int
               'nqx3',                               # pw int
               'nr1',                                # pw/cp int
               'nr1b',                               # cp int
               'nr1s',                               # pw/cp int
               'nr2',                                # pw/cp int
               'nr2b',                               # cp int
               'nr2s',                               # pw/cp int
               'nr3',                                # pw/cp int
               'nr3b',                               # cp int
               'nr3s',                               # pw/cp int
               'nspin',                              # pw/cp int
               'ntyp',                               # pw/cp int
               'occupations',                        # pw/cp str
               'one_atom_occupations',               # pw bool
               'origin_choice',                      # pw int
               'q2sigma',                            # pw/cp float
               'qcutz',                              # pw/cp float
               'relaxz',                             # pw bool
               'report',                             # pw int
               'rhombohedral',                       # pw bool
               'screening_parameter',                # pw float
               'smearing',                           # pw/cp str
               'space_group',                        # pw int
               'starting_charge',                    # pw
               'starting_magnetization',             # pw
               'starting_ns_eigenvalue(m,ispin,I)',  # pw float
               'starting_spin_angle',                # pw bool
               'tot_charge',                         # pw/cp float
               'tot_magnetization',                  # pw/cp float
               'ts_vdw',                             # cp bool
               'ts_vdw_econv_thr',                   # pw/cp float
               'ts_vdw_isolated',                    # pw/cp bool
               'uniqueb',                            # pw bool
               'use_all_frac',                       # pw bool
               'vdw_corr',                           # pw/cp str
               'x_gamma_extrapolation',              # pw bool
               'xdm',                                # pw bool
               'xdm_a1',                             # pw float
               'xdm_a2',                             # pw float
               'zgate',                              # pw float
               'ensemble_energies',                  # pw bool
               'print_ensemble_energies',            # pw bool
               ]

electrons_keys = ['adaptive_thr',  # pw bool
                  'ampre',  # cp float
                  'conv_thr',  # pw/cp float
                  'conv_thr_init',  # pw float
                  'conv_thr_multi',  # pw float
                  'diago_cg_maxiter',  # pw int
                  'diago_david_ndim',  # pw int
                  'diago_full_acc',  # pw bool
                  'diago_rmm_conv',  # pw bool
                  'diago_thr_init',  # pw float
                  'diagonalization',  # pw str
                  'efield',  # pw/cp float
                  'efield_cart',  # pw
                  'efield_phase',  # pw str
                  'ekincw',  # cp float
                  'electron_damping',  # cp float
                  'electron_dynamics',  # cp str
                  'electron_maxstep',  # pw/cp int
                  'electron_temperature',  # cp str
                  'electron_velocities',  # cp str
                  'emass',  # cp float
                  'emass_cutoff',  # cp float
                  'epol',  # cp int
                  'fnosee',  # cp float
                  'grease',  # cp float
                  'lambda_cold',  # cp float
                  'maxiter',  # cp int
                  'mixing_beta',  # pw float
                  'mixing_fixed_ns',  # pw int
                  'mixing_mode',  # pw str
                  'mixing_ndim',  # pw int
                  'n_inner',  # cp int
                  'ninter_cold_restart',  # cp int
                  'niter_cg_restart',  # cp int
                  'ortho_eps',  # cp float
                  'ortho_max',  # cp int
                  'ortho_para',  # pw/cp int
                  'orthogonalization',  # cp str
                  'passop',  # cp float
                  'scf_must_converge',  # pw bool
                  'startingpot',  # pw str
                  'startingwfc',  # pw/cp str
                  'tcg',  # cp bool
                  'tqr',  # pw bool
                  ]

ions_keys = ['amprp',              # cp
             'bfgs_ndim',          # pw int
             'delta_t',            # pw float
             'fnhscl',             # cp
             'fnosep',             # cp float
             'greasp',             # cp float
             'iesr',               # cp int
             'ion_damping',        # cp float
             'ion_dynamics',       # pw/cp str
             'ion_nstepe',         # cp int
             'ion_positions',      # pw/cp str
             'ion_radius',         # cp
             'ion_temperature',    # pw/cp str
             'ion_velocities',     # cp str
             'ndega',              # cp int
             'nhgrp',              # cp
             'nhpcl',              # cp int
             'nhptyp',             # cp int
             'nraise',             # pw int
             'pot_extrapolation',  # pw str
             'refold_pos',         # pw bool
             'remove_rigid_rot',   # pw/cp bool
             'tempw',              # pw/cp float
             'tolp',               # pw/cp float
             'tranp',              # cp
             'trust_radius_ini',   # pw float
             'trust_radius_max',   # pw float
             'trust_radius_min',   # pw float
             'upscale',            # pw float
             'w_1',                # pw float
             'w_2',                # pw float
             'wfc_extrapolation',  # pw str
             ]

rism_keys = ['closure',                # pw str
             'ecutsolv',               # pw float
             'laue_both_hands',        # pw bool
             'laue_buffer_left',       # pw float
             'laue_buffer_right',      # pw float
             'laue_expand_left',       # pw float
             'laue_expand_right',      # pw float
             'laue_nfit',              # pw int
             'laue_reference',         # pw str
             'laue_starting_left',     # pw float
             'laue_starting_right',    # pw float
             'laue_wall',              # pw str
             'laue_wall_epsilon',      # pw float
             'laue_wall_lj6',          # pw bool
             'laue_wall_rho',          # pw float
             'laue_wall_sigma',        # pw float
             'laue_wall_z',            # pw float
             'mdiis1d_size',           # pw int
             'mdiis1d_step',           # pw float
             'mdiis3d_size',           # pw int
             'mdiis3d_step',           # pw float
             'nsolv',                  # pw int
             'permittivity',           # pw float
             'rism1d_bond_width',      # pw float
             'rism1d_conv_thr',        # pw float
             'rism1d_maxstep',         # pw int
             'rism1d_nproc',           # pw int
             'rism1d_nproc_switch',    # pw int
             'rism3d_conv_level',      # pw float
             'rism3d_conv_thr',        # pw float
             'rism3d_maxstep',         # pw int
             'rism3d_planar_average',  # pw bool
             'rmax1d',                 # pw float
             'rmax_lj',                # pw float
             'smear1d',                # pw float
             'smear3d',                # pw float
             'solute_epsilon',         # pw
             'solute_lj',              # pw
             'solute_sigma',           # pw
             'starting1d',             # pw str
             'starting3d',             # pw str
             'tempv',                  # pw float
             'solvents',               # entry of MOL files
             'cations',                # entry of MOL files
             'anions',                 # entry of MOL files
             ]

fcp_keys = ['fcp_conv_thr',      # pw float
            'fcp_delta_t',       # pw float
            'fcp_dynamics',      # pw str
            'fcp_mass',          # pw float
            'fcp_mu',            # pw float
            'fcp_ndiis',         # pw int
            'fcp_nraise',        # pw int
            'fcp_rdiis',         # pw float
            'fcp_temperature',   # pw str
            'fcp_tempw',         # pw float
            'fcp_tolp',          # pw float
            'fcp_velocity',      # pw float
            'freeze_all_atoms',  # pw bool
            'fermilevel']

cell_keys = ['cell_damping',      # cp float
             'cell_dofree',       # pw/cp str
             'cell_dynamics',     # pw/cp str
             'cell_factor',       # pw/cp float
             'cell_parameters',   # cp str
             'cell_temperature',  # cp str
             'cell_velocities',   # cp str
             'fnoseh',            # cp float
             'greash',            # cp float
             'press',             # pw/cp float
             'press_conv_thr',    # pw float
             'temph',             # cp float
             'wmass',             # pw/cp float
             ]

wannier_keys = ['adapt',             # cp bool
                'calwf',             # cp int
                'efx0',              # cp float
                'efx1',              # cp float
                'efy0',              # cp float
                'efy1',              # cp float
                'efz0',              # cp float
                'efz1',              # cp float
                'exx_dis_cutoff',    # cp float
                'exx_me_rcut_pair',  # cp float
                'exx_me_rcut_self',  # cp float
                'exx_neigh',         # cp int
                'exx_poisson_eps',   # cp float
                'exx_ps_rcut_pair',  # cp float
                'exx_ps_rcut_self',  # cp float
                'maxwfdt',           # cp float
                'nit',               # cp int
                'nsd',               # cp int
                'nsteps',            # cp int
                'nwf',               # cp int
                'sw_len',            # cp int
                'tolw',              # cp float
                'wf_efield',         # cp bool
                'wf_friction',       # cp float
                'wf_q',              # cp float
                'wf_switch',         # cp bool
                'wfdt',              # cp float
                'wffort',            # cp int
                'wfsd',              # cp int
                'writev',            # cp bool
                ]

press_ai_keys = ['P_ext',    # cp float
                 'P_fin',    # cp float
                 'P_in',     # cp float
                 'Surf_t',   # cp float
                 'abivol',   # cp bool
                 'abivol',   # cp bool
                 'dthr',     # cp float
                 'pvar',     # cp bool
                 'rho_thr',  # cp float
                 ]

# Data types
inputs_string = ['U_projection_type',
                 'assume_isolated',
                 'calculation',
                 'cell_dofree',
                 'cell_dynamics',
                 'cell_parameters',
                 'cell_temperature',
                 'cell_velocities',
                 'constr_type',
                 'constrained_magnetization',
                 'diagonalization',
                 'disk_io',
                 'efield_phase',
                 'electron_dynamics',
                 'electron_temperature',
                 'electron_velocities',
                 'esm_bc',
                 'exxdiv_treatment',
                 'input_dft',
                 'ion_dynamics',
                 'ion_positions',
                 'ion_temperature',
                 'ion_velocities',
                 'memory',
                 'mixing_mode',
                 'occupations',
                 'orthogonalization',
                 'outdir',
                 'pot_extrapolation',
                 'prefix',
                 'pseudo_dir',
                 'restart_mode',
                 'smearing',
                 'startingpot',
                 'startingwfc',
                 'title',
                 'vdw_corr',
                 'verbosity',
                 'wfc_extrapolation',
                 'wfcdir',
                 'closure',
                 'closure_allowed(2)',
                 'fcp_dynamics',
                 'fcp_dynamics_allowed(7)',
                 'fcp_temperature',
                 'laue_reference',
                 'laue_reference_allowed(4)',
                 'laue_wall',
                 'laue_wall_allowed(3)',
                 'solute_lj(i)',
                 'solute_lj_allowed(4)',
                 'starting1d',
                 'starting1d_allowed(3)',
                 'starting3d',
                 'starting3d_allowed(2)',
                 ]

inputs_float = ['A,',
                'Hubbard_J0(i),',
                'Hubbard_U(i),',
                'Hubbard_alpha(i),',
                'Hubbard_beta(i),',
                'Mass_X',
                'P_ext',
                'P_fin',
                'P_in',
                'Surf_t',
                'ampre',
                'amprp(i),',
                'angle1(i),',
                'angle2(i),',
                'block_1',
                'block_2',
                'block_height',
                'cell_damping',
                'cell_factor',
                'celldm(i),',
                'constr_target',
                'constr_tol',
                'conv_thr',
                'conv_thr_init',
                'conv_thr_multi',
                'degauss',
                'delta_t',
                'diago_thr_init',
                'dt',
                'dthr',
                'eamp',
                'ecfixed',
                'ecutfock',
                'ecutrho',
                'ecutvcut',
                'ecutwfc',
                'efield',
                'efield_cart(i),',
                'efx0,',
                'efx0',
                'efy0',
                'efz0',
                'efx1,',
                'efx1',
                'efy1',
                'efz1',
                'ekin_conv_thr',
                'ekincw',
                'electron_damping',
                'emass',
                'emass_cutoff',
                'emaxpos',
                'eopreg',
                'esm_efield',
                'esm_w',
                'etot_conv_thr',
                'exx_dis_cutoff',
                'exx_fraction',
                'exx_me_rcut_pair',
                'exx_me_rcut_self',
                'exx_poisson_eps',
                'exx_ps_rcut_pair',
                'exx_ps_rcut_self',
                'f_inp1',
                'f_inp2',
                'fcp_mu',
                'fixed_magnetization(i),',
                'fnhscl(i),',
                'fnosee',
                'fnoseh',
                'fnosep',
                'forc_conv_thr',
                'fx',
                'fy',
                'fz',
                'grease',
                'greash',
                'greasp',
                'ion_damping',
                'ion_radius(i),',
                'lambda',
                'lambda_cold',
                'london_c6(i),',
                'london_rcut',
                'london_rcut',
                'london_rvdw(i),',
                'london_s6',
                'london_s6',
                'max_seconds',
                'maxwfdt',
                'mixing_beta',
                'ortho_eps',
                'passop',
                'press',
                'press_conv_thr',
                'q2sigma',
                'qcutz',
                'rho_thr',
                'screening_parameter',
                'starting_charge(i),',
                'starting_magnetization(i),',
                'starting_ns_eigenvalue(m,ispin,I)',
                'temph',
                'tempw',
                'tolp',
                'tolw',
                'tot_charge',
                'tot_magnetization',
                'trust_radius_ini',
                'trust_radius_max',
                'trust_radius_min',
                'ts_vdw_econv_thr',
                'ts_vdw_econv_thr',
                'upscale',
                'v1',
                'v2',
                'v3',
                'vx',
                'vy',
                'vz',
                'w_1',
                'w_2',
                'wf_friction',
                'wf_q',
                'wfdt',
                'wmass',
                'xdm_a1',
                'xdm_a2',
                'xk_x,',
                'xk_x',
                'xk_y',
                'xk_z',
                'wk',
                'zgate',
                'ecutsolv',
                'esm_a',
                'esm_efield',
                'esm_w',
                'esm_zb',
                'fcp_conv_thr',
                'fcp_delta_t',
                'fcp_mass',
                'fcp_mu',
                'fcp_rdiis',
                'fcp_tempw',
                'fcp_tolp',
                'fcp_velocity',
                'laue_buffer_left',
                'laue_buffer_right',
                'laue_expand_left',
                'laue_expand_right',
                'laue_starting_left',
                'laue_starting_right',
                'laue_wall_epsilon',
                'laue_wall_rho',
                'laue_wall_sigma',
                'laue_wall_z',
                'mdiis1d_step',
                'mdiis3d_step',
                'permittivity',
                'rism1d_bond_width',
                'rism1d_conv_thr',
                'rism3d_conv_level',
                'rism3d_conv_thr',
                'rmax1d',
                'rmax_lj',
                'smear1d',
                'smear3d',
                # 'solute_epsilon(i)',
                # 'solute_sigma(i)',
                'tempv',
                ]

inputs_bool = ['abivol',
               'adapt',
               'adaptive_thr',
               'block',
               'diago_full_acc',
               'diago_rmm_conv'
               'dipfield',
               'ensemble_energies',
               'force_symmorphic',
               'gate',
               'lberry',
               'lda_plus_u',
               'lelfield',
               'lfcpopt',
               'lkpoint_dir',
               'london',
               'lorbm',
               'lspinorb',
               'no_t_rev',
               'noinv',
               'noncolin',
               'nosym',
               'nosym_evc',
               'one_atom_occupations',
               'print_ensemble_energies',
               'pvar',
               'refold_pos',
               'relaxz',
               'remove_rigid_rot',
               'rhombohedral',
               'saverho',
               'scf_must_converge',
               'starting_spin_angle',
               'tabps',
               'tcg',
               'tefield',
               'tprnfor',
               'tqr',
               'tranp(i),',
               'ts_vdw',
               'ts_vdw_isolated',
               'tstress',
               'uniqueb',
               'use_all_frac',
               'wf_collect',
               'wf_efield',
               'wf_switch',
               'writev',
               'x_gamma_extrapolation',
               'xdm',
               'tqmmm',
               'trism',
               'esm_debug',
               'freeze_all_atoms',
               'laue_both_hands',
               'laue_wall_lj6',
               'rism3d_planar_average',
               ]

inputs_int = ['bfgs_ndim',
              'calwf',
              'diago_cg_maxiter',
              'diago_david_ndim',
              'edir',
              'electron_maxstep',
              'electron_maxstep',
              'epol',
              'esm_nfit',
              'exx_neigh',
              'gdir',
              'ibrav',
              'iesr',
              'if_pos(1),',
              'ion_nstepe',
              'iprint',
              'isave',
              'iwf',
              'lda_plus_u_kind',
              'maxiter',
              'mixing_fixed_ns',
              'mixing_ndim',
              'n_inner',
              'nat',
              'nat',
              'nberrycyc',
              'nbnd',
              'nconstr',
              'ndega',
              'ndr',
              'ndw',
              'nhgrp(i),',
              'nhpcl',
              'nhptyp',
              'ninter_cold_restart',
              'nit',
              'niter_cg_restart',
              'nk1,',
              'nk1',
              'nk2',
              'nk3',
              'nks',
              'nppstr',
              'nqx1',
              'nqx2',
              'nqx3',
              'nr1',
              'nr2',
              'nr3',
              'nr1b',
              'nr2b',
              'nr3b',
              'nr1s',
              'nr2s',
              'nr3s',
              'nraise',
              'nsd',
              'nspin',
              'nstep',
              'nsteps',
              'ntyp',
              'ntyp',
              'nwf',
              'origin_choice',
              'ortho_max',
              'ortho_para',
              'report',
              'sk1',
              'sk2',
              'sk3',
              'sk1,',
              'space_group',
              'sw_len',
              'wffort',
              'wfsd',
              'esm_debug_gpmax',
              'esm_nfit',
              'fcp_ndiis',
              'fcp_nraise',
              'laue_nfit',
              'mdiis1d_size',
              'mdiis3d_size',
              'nsolv',
              'rism1d_maxstep',
              'rism1d_nproc',
              'rism1d_nproc_switch',
              'rism3d_maxstep',
              ]

inputs_array = []

# dictionary defined all inputs for each namelist dissected into data type
d_input_types = {
    'control': {
        'str': [
            'calculation',
            'disk_io',
            'memory',
            'outdir',
            'prefix',
            'pseudo_dir',
            'restart_mode',
            'title',
            'verbosity',
            'wfcdir',
        ],
        'float': [
            'dt',
            'ekin_conv_thr',
            'etot_conv_thr',
            'forc_conv_thr',
            'max_seconds',
        ],
        'int': [
            'gdir',
            'iprint',
            'isave',
            'nberrycyc',
            'ndr',
            'ndw',
            'nppstr',
            'nstep',
        ],
        'bool': [
            'dipfield',
            'gate',
            'lberry',
            'lelfield',
            'lfcpopt',
            'lkpoint_dir',
            'lorbm',
            'saverho',
            'tabps',
            'tefield',
            'tprnfor',
            'tstress',
            'wf_collect',
        ], },
    'ions': {
        'str': [
            'ion_dynamics',
            'ion_positions',
            'ion_temperature',
            'ion_velocities',
            'pot_extrapolation',
            'wfc_extrapolation',
        ],
        'float': [
            'delta_t',
            'fnosep',
            'greasp',
            'ion_damping',
            'tempw',
            'tolp',
            'trust_radius_ini',
            'trust_radius_max',
            'trust_radius_min',
            'upscale',
            'w_1',
            'w_2',
        ],
        'int': [
            'bfgs_ndim',
            'iesr',
            'ion_nstepe',
            'ndega',
            'nhpcl',
            'nhptyp',
            'nraise',
        ],
        'bool': [
            'refold_pos',
            'remove_rigid_rot',
        ], },
    'rism': {
        'str': [
            'closure',
            'laue_reference',
            'laue_wall',
            'solute_lj',  # 'solute_lj(1)'
            'starting1d',
            'starting3d',
        ],
        'float': [
            'ecutsolv',
            'laue_buffer_left',
            'laue_buffer_right',
            'laue_expand_left',
            'laue_expand_right',
            'laue_starting_left',
            'laue_starting_right',
            'laue_wall_epsilon',
            'laue_wall_rho',
            'laue_wall_sigma',
            'laue_wall_z',
            'mdiis1d_step',
            'mdiis3d_step',
            'permittivity',
            'rism1d_bond_width',
            'rism1d_conv_thr',
            'rism3d_conv_level',
            'rism3d_conv_thr',
            'rmax1d',
            'rmax_lj',
            'smear1d',
            'smear3d',
            'solute_epsilon',  # 'solute_epsilon(1)'
            'solute_sigma',  # 'solute_sigma(1)',
            'tempv',
        ],
        'int': [
            'laue_nfit',
            'mdiis1d_size',
            'mdiis3d_size',
            'nsolv',
            'rism1d_maxstep',
            'rism1d_nproc',
            'rism1d_nproc_switch',
            'rism3d_maxstep',
        ],
        'bool': [
            'laue_both_hands',
            'laue_wall_lj6',
            'rism3d_planar_average',
        ], },
    'fcp': {
        'str': [
            'fcp_dynamics',
            'fcp_temperature',
        ],
        'float': [
            'fcp_conv_thr',
            'fcp_delta_t',
            'fcp_mass',
            'fcp_mu',
            'fcp_rdiis',
            'fcp_tempw',
            'fcp_tolp',
            'fcp_velocity',
        ],
        'int': [
            'fcp_ndiis',
            'fcp_nraise',
        ],
        'bool': [
            'freeze_all_atoms',
        ], },
    'press_ai': {
        'str': [
        ],
        'float': [
            'P_ext',
            'P_fin',
            'P_in',
            'Surf_t',
            'dthr',
            'rho_thr',
        ],
        'int': [
        ],
        'bool': [
            'abivol',
            'abivol',
            'pvar',
        ], },
    'system': {
        'str': [
            'U_projection_type',
            'assume_isolated',
            'constrained_magnetization',
            'esm_bc',
            'exxdiv_treatment',
            'input_dft',
            'occupations',
            'smearing',
            'vdw_corr',
        ],
        'float': [
            'block_1',
            'block_2',
            'block_height',
            'degauss',
            'eamp',
            'ecfixed',
            'ecutfock',
            'ecutrho',
            'ecutvcut',
            'ecutwfc',
            'emaxpos',
            'eopreg',
            'esm_efield',
            'esm_w',
            'exx_fraction',
            'fcp_mu',
            'lambda',
            'london_rcut',
            'london_s6',
            'q2sigma',
            'qcutz',
            'screening_parameter',
            'starting_ns_eigenvalue(m,ispin,I)',
            'tot_charge',
            'tot_magnetization',
            'ts_vdw_econv_thr',
            'xdm_a1',
            'xdm_a2',
            'zgate',
        ],
        'int': [
            'edir',
            'esm_nfit',
            'ibrav',
            'lda_plus_u_kind',
            'nat',
            'nbnd',
            'nqx1',
            'nqx2',
            'nqx3',
            'nr1',
            'nr1b',
            'nr1s',
            'nr2',
            'nr2b',
            'nr2s',
            'nr3',
            'nr3b',
            'nr3s',
            'nspin',
            'ntyp',
            'origin_choice',
            'report',
            'space_group',
        ],
        'bool': [
            'block',
            'ensemble_energies',
            'force_symmorphic',
            'lda_plus_u',
            'london',
            'lspinorb',
            'no_t_rev',
            'noinv',
            'noncolin',
            'nosym',
            'nosym_evc',
            'one_atom_occupations',
            'print_ensemble_energies',
            'relaxz',
            'rhombohedral',
            'starting_spin_angle',
            'ts_vdw',
            'ts_vdw_isolated',
            'uniqueb',
            'use_all_frac',
            'x_gamma_extrapolation',
            'xdm',
        ], },
    'cell': {
        'str': [
            'cell_dofree',
            'cell_dynamics',
            'cell_parameters',
            'cell_temperature',
            'cell_velocities',
        ],
        'float': [
            'cell_damping',
            'cell_factor',
            'fnoseh',
            'greash',
            'press',
            'press_conv_thr',
            'temph',
            'wmass',
        ],
        'int': [
        ],
        'bool': [
        ], },
    'electrons': {
        'str': [
            'diagonalization',
            'efield_phase',
            'electron_dynamics',
            'electron_temperature',
            'electron_velocities',
            'mixing_mode',
            'orthogonalization',
            'startingpot',
            'startingwfc',
        ],
        'float': [
            'ampre',
            'conv_thr',
            'conv_thr_init',
            'conv_thr_multi',
            'diago_thr_init',
            'efield',
            'ekincw',
            'electron_damping',
            'emass',
            'emass_cutoff',
            'fnosee',
            'grease',
            'lambda_cold',
            'mixing_beta',
            'ortho_eps',
            'passop',
        ],
        'int': [
            'diago_cg_maxiter',
            'diago_david_ndim',
            'electron_maxstep',
            'epol',
            'maxiter',
            'mixing_fixed_ns',
            'mixing_ndim',
            'n_inner',
            'ninter_cold_restart',
            'niter_cg_restart',
            'ortho_max',
            'ortho_para',
        ],
        'bool': [
            'adaptive_thr',
            'diago_full_acc',
            'diago_rmm_conv',
            'scf_must_converge',
            'tcg',
            'tqr',
        ], },
    'wannier': {
        'str': [
        ],
        'float': [
            'efx0',
            'efx1',
            'efy0',
            'efy1',
            'efz0',
            'efz1',
            'exx_dis_cutoff',
            'exx_me_rcut_pair',
            'exx_me_rcut_self',
            'exx_poisson_eps',
            'exx_ps_rcut_pair',
            'exx_ps_rcut_self',
            'maxwfdt',
            'tolw',
            'wf_friction',
            'wf_q',
            'wfdt',
        ],
        'int': [
            'calwf',
            'exx_neigh',
            'nit',
            'nsd',
            'nsteps',
            'nwf',
            'sw_len',
            'wffort',
            'wfsd',
        ],
        'bool': [
            'adapt',
            'wf_efield',
            'wf_switch',
            'writev',
        ], },
}


# populate mappings from ASE keys to QE keys
# d_keymap = {}
# native_keys = control_keys + system_keys + electrons_keys + ions_keys + rism_keys + cell_keys + wannier_keys + press_ai_keys
# for key in native_keys:
# d_keymap[key] = key

d_custom_keys = {
    'control': {
        'calcmode': 'calculation',
        'timestep': 'dt',
        'ediff': 'etot_conv_thr',
        'fmax': 'forc_conv_thr',
        'psppath': 'pseudo_dir',
        'restart': 'restart_mode',
        'field': 'tefield',
        'printforces': 'tprnfor',
        'calcstress': 'tstress',
        'verbose': 'verbosity',
        'lwave': 'wf_collect',
        'rism': 'trism',
        'constmu': 'lfcp',
        'finfield': 'lelfield',
        'field': 'tefield',
        'dipole': 'dipfield',
    },
    'ions': {
        'ion_refold': 'refold_pos',
        'nhfreq': 'fnosep',
        'temperature': 'tempw',
    },
    'system': {
        'isolated': 'assume_isolated',
        'sigma': 'degauss',
        'fw': 'ecutfock',
        'dw': 'ecutrho',
        'pw': 'ecutwfc',
        'dipole_direction': 'edir',
        'beefensemble': 'ensemble_energies',
        'fix_magmom': 'fixed_magnetization',
        'J': 'Hubbard_J(i,ityp)',
        'U': 'Hubbard_U',
        'U_alpha': 'Hubbard_alpha',
        'xc': 'input_dft',
        'spinorbit': 'lspinorb',
        'nbands': 'nbnd',
        'noncollinear': 'noncolin',
        'usp_grid': 'nr1',
        'fft_grid': 'nr1b',
        'spinpol': 'nspin',  # careful
        'printensemble': 'print_ensemble_energies',
    },
    'cell': {
        'pressure': 'press_conv_thr',
    },
    'electrons': {
        'diag': 'diagonalization',
        'nhtemp_elec': 'ekincw',
        'scf_maxsteps': 'electron_maxstep',
        'nhfreq_elec': 'fnosee',
        'mixing': 'mixing_beta',
        'ortho': 'orthogonalization',
    },
    'wannier': {
        'nwann_tot_iter': 'nit',
        'nwann_sd_iter': 'nsd',
        'nwannsteps': 'nsteps',
        'wann_tolerance': 'tolw',
    },
    'rism': {
        'LJ': 'solute_lj',
        'LJ_epsilons': 'solute_epsilons',
        'LJ_sigmas': 'solute_sigmas',
    },
    'fcp': {
        'fermilevel': 'fcp_mu',
    },
    'press_ai': {},
    'kpoints': {},
}


class cpespresso(Calculator):
    """
    hacked ase interface for Quantum Espresso:
    restarts from  outdir/prefix_ndr
    saves files as outdir/prefix_ndw
    """
    name = 'espresso'

    def __init__(
            self,
            atoms=None,
            setups=None,
            pwinputfile='pw.in',
            coordunits='angstrom',
            code='cp',
            **kwargs):
        """ Initalization of all QE input settings
        """
        # define namelists
        self.params_control = {}
        self.params_system = {}
        self.params_electron = {}
        self.params_ion = {}
        self.params_rism = {}
        self.params_fcp = {}
        self.params_wannier = {}
        self.params_pressai = {}
        self.params_cell = {}
        self.params_kpts = {}
        self.params_convergence = {}
        self.masses = {}

        # define all namelists keys
        self._populate(control_keys, self.params_control, 'control')
        self._populate(system_keys, self.params_system, 'system')
        self._populate(electrons_keys, self.params_electron, 'electrons')
        self._populate(ions_keys, self.params_ion, 'ions')
        self._populate(rism_keys, self.params_rism, 'rism')
        self._populate(fcp_keys, self.params_fcp, 'fcp')
        self._populate(wannier_keys, self.params_wannier, 'wannier')
        self._populate(press_ai_keys, self.params_pressai, 'press_ai')
        self._populate(cell_keys, self.params_cell, 'cell')
        self._populate(kpoint_keys, self.params_kpts, 'kpoints')

        # define custom ASE key mappings to QE keys
        self._populate(
            d_custom_keys['control'].keys(),
            self.params_control,
            'control')
        self._populate(
            d_custom_keys['system'].keys(),
            self.params_system,
            'system')
        self._populate(
            d_custom_keys['electrons'].keys(),
            self.params_electron,
            'electrons')
        self._populate(d_custom_keys['ions'].keys(), self.params_ion, 'ions')
        self._populate(d_custom_keys['rism'].keys(), self.params_rism, 'rism')
        self._populate(d_custom_keys['fcp'].keys(), self.params_fcp, 'fcp')
        self._populate(
            d_custom_keys['wannier'].keys(),
            self.params_wannier,
            'wannier')
        self._populate(
            d_custom_keys['press_ai'].keys(),
            self.params_pressai,
            'press_ai')
        self._populate(d_custom_keys['cell'].keys(), self.params_cell, 'cell')
        self._populate(
            d_custom_keys['kpoints'].keys(),
            self.params_kpts,
            'kpoints')

        for key in convergence_keys:
            self.params_convergence[key] = None

        #######################################################################
        self.defaults_cp = {
            'deuterate': 1,
            ###################################################################
            'kpts': (1, 1, 1),
            'kptshift': (0, 0, 0),

            'pwinputfile': 'pw.in',
            'calcmode': 'cp',
            'calculation': 'cp',
            'isave': 10,
            'iprint': 1,
            'timestep': 8,
            'outdir': './cpcalc',
            'prefix': 'calc',
            'ndr': 50,
            'ndw': 51,
            #         'opt_algorithm' : 'bfgs',   # this needs to be reconciled with ion_dynamics
            ##################################################################################
            # DEBUG: OUTPUT FLAGS?
            'calcstress': False,
            'printforces': True,        # tprnfor; defaults to True for 'relax', etc.
            # 'verbose'    : 'low',
            # 'disk_io'    : 'default',   # how often espresso writes wavefunctions to disk
            #         'wf_collect' : False,
            #         'removewf'   : False,        # currently not implemented to do anything (deprecated from SUNCAT)
            #         'dipole'    : False,
            #         'field'     : False,  # tefield  ; saw-tooth potential , works for cp & pw
            #         'finfield'  : False,  # lelfield ; finite electric field (modern theory of polarization)
            #         'dipole'    : {'status' : False},
            #         'field'     : {'status' : False},  # tefield  ; saw-tooth potential , works for cp & pw
            #         'finfield'  : {'status' : False},  # lelfield ; finite electric field (modern theory of polarization)


            # DEBUG: CONVERGENCE FLAGS?
            'convergence': None,  # simply to include the debug flags explicitly in input file or not (to be easily modified if desired)
            'etot_conv_thr': 2e-3,  # eV,   [control]   etot_conv_thr ; total/ionic energy convergence criteria (QE default 1E-4 Ry=1E-3 eV, EDIFFG in VASP default 1E-3)
            'fmax': 2e-2,  # eV/A, [control]   forc_conv_thr ; maximum force tolerance defaults to 10*total energy convergence criteria; both must be satisfied
            'ekin_conv_thr': None,  # eV,   [control]   ekin_conv_thr ; cp only, QE defaults to 1E-6 Ry?Ha? internally
            'elec_conv_thr': None,  # eV,   [electrons] conv_thr ; electronic convergence self-consistency criteria (QE default 1E-6 in their units, EDIFF in VASP default 1E-4)
            'scf_maxsteps': 500,  # [electrons] electron_maxstep    maximum number of iterations in a scf step
            'mixing': 0.7,  # [electrons] mixing_beta , internally defaults to 0.7
            'diag': 'david',  # [electrons] diagonalization     'cg' slower but less memory intensive
            'ortho': 'ortho',  # [electrons] ortho ; cp only, electronic orthonormalization method; 'ortho' default
            'pressure': None,  # [cell] press_conv_thr  ; relevant for vc-relax, internal default to 0.5 kbar
            'constraints_tol': None,  # [constraints] constr_tol   Constraints tolerance

            #         'output'    : {'disk_io'   : 'default',    # how often espresso writes wavefunctions to disk
            #                        'avoidio'   : False,        # will overwrite disk_io parameter if True
            #                        'removewf'  : True,
            #                        'removesave': False,
            #                        'wf_collect': False,
            #                        'printforces': False,       # tprnfor
            #                              } ,
            #         'convergence': {'total energy'      : 1e-4, # eV,   [control]   etot_conv_thr ; total/ionic energy convergence criteria (QE default 1E-3, EDIFFG in VASP default 1E-3)
            #                         'fmax'              : None, # eV/A, [control]   forc_conv_thr ; maximum force tolerance defaults to 10*total energy convergence criteria; both must be satisfied
            #                         'kinetic energy'    : None, # eV,   [control]   ekin_conv_thr ; cp only, QE defaults to 1E-5 Ry?Ha? internally
            #                         'electronic energy' : 1e-6, # eV,   [electrons] conv_thr ; electronic convergence self-consistency criteria (QE default 1E-5, EDIFF in VASP default 1E-4)
            #                         'scf_maxsteps'      : 500,  #       [electrons] electron_maxstep    maximum number of iterations in a scf step
            #                         'mixing'            : 0.7,  #       [electrons] mixing_beta , internally defaults to 0.7
            #                         'diag'              : 'david', #    [electrons] diagonalization     'cg' slower but less memory intensive
            #                         'ortho'             : 'ortho', #    [electrons] ortho ; cp only, electronic orthonormalization method; 'ortho' default
            #                         'pressure'          : None,    #    [cell] press_conv_thr  ; relevant for vc-relax, internal default to 0.5 kbar
            #                         'constraints'       : None,    #    [constraints] constr_tol   Constraints tolerance
            #                         },
            ##################################################################################

            'ibrav': 0,        # let ase handle structure
            'pw': 400.0,    # ecutwfc
            'nbands': 0,        # negative value means + extra unoccupied bands

            # auto-populate cp settings
            'usp_grid': (30, 30, 30),
            'emass': 400,  # (a.u. = 1/1822.9 a.mu.u = 9.1E-31 kg) cp only
            'emass_cutoff': 4,    # (Ry) cp only
            'ortho_max': 1000,      # cp only

            'occupations': 'smearing',  # 'smearing', 'fixed', 'tetrahedra', 'from_input'
            'smearing': 'gaussian',  # 'gaussian','gauss', 'fermi-dirac','f-d','fd', 'methfessel-paxton','m-p','mp', marzari-vanderbilt', 'cold', 'm-v', 'mv'
            'sigma': 0.00,        # nonzero sigma initiates smearing

            'tot_magnetization': -1,  # -1 means unspecified, 'hund' means Hund's rule for each atom
            'fix_magmom': False,   # make this compatible with other magnetization flags
            'spinpol': False,
            'noncollinear': False,
            'spinorbit': False,
            'nosym': False,
            'noinv': False,
            'nosym_evc': False,
            'no_t_rev': False,
            'beefensemble': False,
            'printensemble': False,

            'U_projection_type': 'atomic',

            'temperature': 0.,    # tempw, MD temperature (K)
            'ion_refold': False,  # pw_only, refold in atomic positions to supercell

            # Wannier
            ##################################################################################
            'calwf': 4,       # calwf, start from KS states, do 1 CP step then generate wannier functions after nwannsteps of damped dynamics
            'nwannsteps': 15000,   # nsteps,  # of Damped-Dynamics steps per CP step
            'nwann_tot_iter': 600,     # nit,     # of steepest descent + conjugate gradient iterations for Wannier convergence (n_cg iterations = nit - nsd)
            'nwann_sd_iter': 300,     # nsd,     # of steepest descent iterations for Wannier convergence
            # 'wann_tolerance': 1.E-6,    # tolw,    # convergence criteria for localization of Wannier functions (CP default is 1.D-8)
        }
        #######################################################################

        self.defaults_pw = {
            'deuterate': 0,
            'kpts': (1, 1, 1),
            'kptshift': (0, 0, 0),

            'pwinputfile': 'pw.in',
            'calcmode': 'relax',
            'calculation': 'relax',
            'outdir': './cpcalc',
            'prefix': 'calc',
            'restart': 'from_scratch',
            #        'printforces': True,        # tprnfor; defaults to True for 'relax', etc.

            #         'wf_collect' : False,
            #         'dipole'    : False,
            #         'field'     : False,  # tefield  ; saw-tooth potential , works for cp & pw
            #         'finfield'  : False,  # lelfield ; finite electric field (modern theory of polarization)
            'convergence': None,  # simply to include the debug flags explicitly in input file or not (to be easily modified if desired)
            'etot_conv_thr': 2e-3,  # eV,   [control]   etot_conv_thr ; total/ionic energy convergence criteria (QE default 1E-4 Ry=1E-3 eV, EDIFFG in VASP default 1E-3)
            'fmax': 2e-2,  # eV/A, [control]   forc_conv_thr ; maximum force tolerance defaults to 10*total energy convergence criteria; both must be satisfied
            'ibrav': 0,        # let ase handle structure
            'pw': 400.0,    # ecutwfc (eV)
            'ecutwfc': 400.0,    # ecutwfc (eV)
            #         'nbands'    : 0,        # negative value means + extra unoccupied bands
            #         'usp_grid'     : (30,30,30),

            'occupations': 'smearing',  # 'smearing', 'fixed', 'tetrahedra', 'from_input'
            'smearing': 'gaussian',  # 'gaussian','gauss', 'fermi-dirac','f-d','fd', 'methfessel-paxton','m-p','mp', marzari-vanderbilt', 'cold', 'm-v', 'mv'
            'sigma': 0.10,        # eV, nonzero sigma initiates smearing

            'tot_magnetization': -1,  # -1 means unspecified, 'hund' means Hund's rule for each atom
            'edir': 3,
            'eamp': 0.0,
            'eopreg': 0.025,
            #         'emaxpos' : find_max_empty_space(self.atoms,edir),
        }
        #######################################################################

        self.atoms = atoms
        self.setups = setups
        self.coordunits = coordunits

# # DEBUG: KPOINTS
#         if type(kpts)==float or type(kpts)==int:
#             from ase.calculators.calculator import kptdensity2monkhorstpack
#             kpts = kptdensity2monkhorstpack(atoms, kpts)
#         elif isinstance(kpts, StringType):
#             assert kpts == 'gamma'
#         else:
#             assert len(kpts) == 3
#         self.params_kpts['kpts'] = kpts
#         self.params_kpts['kptshift'] = kptshift

# DEBUG: CONVERGENCE
#         self.convergence   = convergence
#         self.conv_thr      = self.convergence['electronic energy'] # standalone mod    # QE conv_thr / VASP EDIFF
#         self.etot_conv_thr = self.convergence['total energy']      # standalone mod    # QE etot_conv_thr / VASP EDIFFG
#         self.ekin_conv_thr = self.convergence['kinetic energy']    # standalone mod    # QE ekin_conv_thr for CP-MD runs
#         self.forc_conv_thr = self.convergence['fmax']              # standalone mod    # QE forc_conv_thr
# self.constr_tol    = self.convergence['constraints']       # standalone
# mode for constraints tolerance : constr_tol

# DEBUG: SMEARING/OCCUPATION
#         if type(smearing)==str:
#             self.smearing = smearing
#             self.sigma    = sigma
#         else:
#             self.smearing = smearing[0]
#             self.sigma    = smearing[1]

        # FIX: add toggle to set given default parameters
        if code == 'cp':
            print("generating cp calculator")
            self.defaults = self.defaults_cp

        elif code == 'rism':
            self.defaults_rism = {
                'trism': True,
                #            'constmu'  : True,
                'isolated': 'esm',
                'esm_bc': 'bc1',
                'tot_charge': 0.0,
                #             'startingwfc' : 'file',
                #             'startingpot' : 'file',
                'nosym': True,
                'mixing_mode': 'local-TF',
                'mixing_beta': 0.2,
                'electron_maxstep': 500,
                'conv_thr': 1.0E-6,
                'diagonalization': 'rmm',
                'diago_rmm_conv': False,
                'starting1d': 'zero',
                'starting3d': 'zero',
                # 'nsolv'      : 3,
                'closure': 'kh',
                'tempv': 300.0,  # ! Kelvin
                'ecutsolv': 300.0,  # ! Rydberg
                'rism1d_conv_thr': 1.0E-8,
                'rism3d_maxstep': 4000,
                'rism3d_conv_thr': 5.0E-5,
                #             'solute_epsilon(1)' : 0.070, # !kcal/mol
                #             'solute_epsilon(2)' : 0.030, # !kcal/mol
                #             'solute_epsilon(3)' : 0.191, # !kcal/mol
                #             'solute_sigma(1)' : 3.55, # !angstrom
                #             'solute_sigma(2)' : 2.42, # !angstrom
                #             'solute_sigma(3)' : 1.46, # !angstrom

                'laue_expand_right': 75.0,  # ! bohr
                # 'laue_starting_right' : 1.5,  # ! bohr
                'laue_starting_right': 3.0,  # ! bohr
                #            'laue_buffer_right'   : 20.0, # ! bohr
            }

#             defaults_rism['trism'] = True
#             defaults_rism['constmu'] = True
#             defaults_rism['isolated'] = True
#             defaults_rism['esm_bc'] = 'bc1'
#             defaults_rism['tot_charge'] = 0.5 # set charge
#             defaults_rism = self.defaults_pw
            for key, val in self.defaults_pw.iteritems():
                self.defaults_rism[key] = val
            print("defaulting pwscf-rism calculator")
            self.defaults = self.defaults_rism
        else:
            print("defaulting pwscf calculator")
            self.defaults = self.defaults_pw

        # auto populate defaults from self.defaults
        self.set(**self.defaults)

        # reset arguments from specified inputs
        self.set(**kwargs)

        if atoms is not None:
            atoms.set_calculator(self)

    def _populate(self, keylist, d_namelist, namelist, value=None):
        """ Initialize the namelists and input flags with input value (defaults to None). Input
            list of keys
            dictionary of custom keys (e.g. self.params_control)
            namelist (e.g. 'control', 'system')
        """
        for key in keylist:
            popd = d_namelist
            refd = d_custom_keys[namelist]

            popd[key] = value
            if key in refd.keys():
                popd[refd[key]] = value

    def set(self, **kwargs):
        """ Define settings for the Quantum Espresso calculator object after it has been initialized.
        This is done in the following way:

        >> calc = espresso(...)
        >> atoms = set.calculator(calc)
        >> calc.set(xc='BEEF')

        NB: No input validation is made
        """
        for key in kwargs:

            if key in self.params_control:
                self.params_control[key] = kwargs[key]
            elif key in self.params_system:
                self.params_system[key] = kwargs[key]
            elif key in self.params_electron:
                self.params_electron[key] = kwargs[key]
            elif key in self.params_ion:
                self.params_ion[key] = kwargs[key]
            elif key in self.params_rism:
                self.params_rism[key] = kwargs[key]
            elif key in self.params_fcp:
                self.params_fcp[key] = kwargs[key]
            elif key in self.params_wannier:
                self.params_wannier[key] = kwargs[key]
            elif key in self.params_cell:
                self.params_cell[key] = kwargs[key]
            elif key in self.params_pressai:
                self.params_pressai[key] = kwargs[key]
            elif key in self.params_kpts:
                self.params_kpts[key] = kwargs[key]
            elif key in self.params_convergence:
                self.params_convergence[key] = kwargs[key]

            # custom keys
            elif key in d_custom_keys['control']:
                self.params_control[key] = kwargs[key]

            elif key in d_custom_keys['system']:
                self.params_system[key] = kwargs[key]

            elif key in d_custom_keys['electrons']:
                self.params_electron[key] = kwargs[key]

            elif key in d_custom_keys['ions']:
                self.params_ion[key] = kwargs[key]

            elif key in d_custom_keys['rism']:
                self.params_rism[key] = kwargs[key]

            elif key in d_custom_keys['fcp']:
                self.params_fcp[key] = kwargs[key]

            elif key in d_custom_keys['wannier']:
                self.params_wannier[key] = kwargs[key]

            elif key in d_custom_keys['cell']:
                self.params_cell[key] = kwargs[key]

            elif key in d_custom_keys['press_ai']:
                self.params_pressai[key] = kwargs[key]

            elif key in d_custom_keys['kpoints']:
                self.params_kpts[key] = kwargs[key]

            # or custom_keys.has_key(key):
            elif key in self.params_convergence:
                self.params_convergence[key] = kwargs[key]

            elif key == 'atoms':
                self.atoms = kwargs[key]
            elif key == 'setups':
                self.setups = kwargs[key]
            elif key == 'pwinputfile':
                self.pwinputfile = kwargs[key]
            elif key == 'deuterate':
                self.deuterate = kwargs[key]

            elif key == 'masses':
                print("mass resetting currently not implemented")
                #self.atoms = kwargs[key]

            else:
                raise TypeError('Parameter not defined: ' + key)

        # Auto create variables from input
        if kwargs == self.defaults:
            print("initializing espresso/cp defaults")
        else:
            print("updating espresso/cp inputs")
            self.input_update()
            self.recalculate = True
        # print("checking inputfile reset: %s" % (self.pwinputfile))

    def input_update(self, debug=0):
        """ Run initialization functions, such that this can be called if
         variables in espresso are changes using set or directly.
        """
        # reset 'relative' coordinates as 'crystal' for QE
        if self.coordunits == 'relative':
            self.coordunits = 'crystal'

        # make density cutoff 10x energy cutoff
        if self.params_system['dw'] is None:
            self.params_system['dw'] = 10. * self.params_system['pw']
        else:
            assert self.params_system['dw'] >= self.params_system['pw']

        # make sure a pseudopotential path is defined
        if self.params_control['psppath'] is None:
            try:
                self.params_control['psppath'] = os.environ['ESP_PSP_PATH']
            except BaseException:
                print('Unable to find pseudopotential path. Consider \
                setting ESP_PSP_PATH environment variable')
                raise

        # make sure BEEF settings are compatible
        if self.params_system['beefensemble']:
            if self.params_system['xc'].upper().find('BEEF') < 0:
                raise KeyError(
                    "ensemble-energies only work with xc=BEEF or variants of it!")

        # make ion_dynamics compatible with other inputs
        if self.params_ion['nhfreq']:
            self.params_ion['nhpcl'] = min([len(self.params_ion['nhfreq']), 4])

        # make RISM nsolv compatible with solvent inputs
        sollist = [
            i for i in [
                self.params_rism['solvents'],
                self.params_rism['cations'],
                self.params_rism['anions']] if i]
        if sollist:
            self.params_rism['nsolv'] = len(flatten(sollist))

##########################################################################

##########################################################################

        if debug:
            print("     DEBUG: be sure to fix how nose frequencies are  handled/reset ")

# DEBUG CONVERGENCE
#         ########### FIX CONVERGENCE TOLERANCE PART #############
#         # set self-consistency convergence for electronic energy (conv_thr / EDIFF)
#         if not self.conv_thr:
#             self.conv_thr = 1e-6/rydberg
#
#         # set force-threshold based on total energy convergence criteria (if not specified)
#         if not self.forc_conv_thr :
#             self.forc_conv_thr  = self.etot_conv_thr*10
#         else:
#             assert self.forc_conv_thr >= self.etot_conv_thr*10
#         #elif self.forc_conv_thr >= self.etot_conv_thr*10:
#         #    print "FORCE TOLERANCE MAY BE TOO LOOSE!"
#        ########### FIX CONVERGENCE TOLERANCE PART #############
#

    def get_version(self):
        return '0.4x'

    def atoms2species(self, verbose=1):
        """ Define several properties of the quantum espresso species from the ase atoms object.
        Takes into account that different spins (or different U etc.) on same kind of
        chemical elements are considered different species in quantum espresso

        Also parses pseudopotential files with parse_pseudopotential to extract relevant info like
         nvalence (total # of electrons)
         psp type (US, NC, PAW)
         ...

        *** FIX : Add nose-hoover freq dictionary here?
        """
        setups = self.setups  # noqa: F841
        symbols = self.atoms.get_chemical_symbols()
        masses = self.atoms.get_masses()
        magmoms = list(self.atoms.get_initial_magnetic_moments())
        charges = _populate_specprop_helper(
            self.params_system['starting_charge'], symbols)

        # Nose-Hoover frequency dictionary of form {'el' : omega}
        nhfreqs = self.params_ion['nhfreq']
        # LJepsilons = self.params_rism['epsilons'] # LJ epsilon parameter for RISM
        # LJsigmas   = self.params_rism['sigmas']   # LJ sigma parameter for
        # RISM

        # NOSE-HOOVER PARAMETERS
        # in case input NH frequencies are not a dictionary, makes them one,
        # but ordering may be screwy
        if nhfreqs:
            if not isinstance(nhfreqs, dict):
                d_nhf = {}
                atypes = list(set(symbols))
                for i, nhf in enumerate(nhfreqs):
                    if i <= len(atypes):
                        d_nhf[atypes[i]] = nhf
                nhfreqs = d_nhf
                print(
                    "*** Created dictionary from list of Nose-Hoover frequencies. Ordering may be incorrect! ***")
                verbose_nh = 1
                self.set(nhfreq=d_nhf)
            else:
                verbose_nh = 0
        else:
            # print "NH frequencies not defined!"
            d_nhf = {}
            atypes = list(set(symbols))
            for atype in atypes:
                d_nhf[atype] = 0.0
            nhfreqs = d_nhf
            verbose_nh = 0
        # END NOSE-HOOVER PARAMETERS

        #######################################################################
        # Define RISM LJ PARAMETERS for each defined species (can account for Co1 vs Co2)
        # solute_lj(i) can be 'uff', 'opls-aa', or 'none' and with solute_epsilon(i) and solute_sigma(i) defined
        #LJtype        = _populate_specprop_helper(self.params_rism['solute_lj'], symbols)
        LJtype = _populate_specprop_helper(
            self.params_rism['solute_lj'], symbols, rawval=['opls-aa'])
        LJepsilonlist = _populate_specprop_helper(
            self.params_rism['solute_epsilons'], symbols)
        LJsigmalist = _populate_specprop_helper(
            self.params_rism['solute_sigmas'], symbols)
        #######################################################################

        if len(magmoms) < len(symbols):
            magmoms += list(np.zeros(len(symbols) - len(magmoms), np.float))

        if self.coordunits == 'crystal' or self.coordunits == 'relative':
            pos = self.atoms.get_scaled_positions()
        elif self.coordunits == 'angstrom':
            pos = self.atoms.get_positions()
        if verbose:
            print("assuming %s atomic coordinates" % (self.coordunits))

        #######################################################################
        # Define Hubbard U and J for each defined species (can account for Co1
        # vs Co2)
        Ulist = _populate_specprop_helper(self.params_system['U'], symbols)
        Jlist = _populate_specprop_helper(self.params_system['J'], symbols)
        U_alphalist = _populate_specprop_helper(
            self.params_system['U_alpha'], symbols)
        #######################################################################

        self.species = []  # species labels
        self.specprops = []  # dictionary of species properties
        dic = {}
        symcounter = {}
        for s in symbols:
            symcounter[s] = 0
        for i, symbol in enumerate(symbols):    # loop over number of atoms
            key = '%s_m%.14eU%.14eJ%.14eUa%.14e' % (
                symbol, magmoms[i], Ulist[i], Jlist[i], U_alphalist[i])
            if key in dic:
                self.specprops.append((dic[key][1], pos[i]))
            else:
                symcounter[symbol] += 1
                specie_id = symbol + str(symcounter[symbol])
                dic[key] = [i, specie_id]
                self.species.append(specie_id)
                self.specprops.append((specie_id, pos[i]))

        self.params_system['nat'] = len(self.atoms)
        self.params_system['ntyp'] = len(self.species)
        self.specdict = {}
        for i, s in dic.values():
            self.specdict[s] = specobj(s=s.strip('0123456789'),  # chemical symbol w/o index
                                       mass=masses[i],
                                       magmom=magmoms[i],
                                       U=Ulist[i],
                                       J=Jlist[i],
                                       U_alpha=U_alphalist[i],
                                       # Nose-Hoover frequency
                                       nosefreq=nhfreqs[s.strip('0123456789')],
                                       LJ_type=LJtype[i],
                                       LJ_epsilon=LJepsilonlist[i],
                                       LJ_sigma=LJsigmalist[i],
                                       qstart=charges[i]
                                       )
            if verbose_nh:
                print("atoms2species taking species %s with Nose-Hoover freq: %s" %
                      (s, nhfreqs[s.strip('0123456789')]))
        self.parse_pseudopotential(verbose=verbose)

    def check_spinpol(self, sigma_tol=1e-13):
        """ For dealing with spin/magnetism in PW.
        Relic from old ase interface.
        """
        mm = self.atoms.get_initial_magnetic_moments()
        sp = mm.any()
        self.summed_magmoms = np.sum(mm)
        if sp:
            if not self.params_system['spinpol'] and not self.params_system['noncollinear']:
                #raise KeyError('Explicitly specify spinpol=True or noncollinear=True for spin-polarized systems')
                print(
                    '*** Explicitly specify spinpol=True or noncollinear=True for spin-polarized systems ***')
            elif abs(self.params_system['sigma']) <= sigma_tol and not self.params_system['fix_magmom']:
                #raise KeyError('Please use fix_magmom=True for sigma=0.0 eV and spinpol=True. Hopefully this is not an extended system...?')
                print('*** Please use fix_magmom=True for sigma=0.0 eV and spinpol=True. Hopefully this is not an extended system...? ***')

        else:
            if self.params_system['spinpol'] and abs(
                    self.params_system['sigma']) <= sigma_tol:
                self.params_system['fix_magmom'] = True
        if abs(self.params_system['sigma']) <= sigma_tol:
            self.params_system['occupations'] = 'fixed'

    def get_species_order(self):
        """ Get the correct ordering of species to be defined in ATOMIC_SPECIES.
        This gives the ordering of the pseudopotentials and also of the Nose-Hoover freqs.
        """
        self.species_order = []

        # first go through and handle the ultrasofts (they must precede
        # norm-conserving/PAW in cp)
        for specie in self.species:
            spec = self.specdict[specie]

            # include species if setups are USP (they default to USP if they
            # arent specified)
            if self.setups and spec.s in self.setups.keys():
                if 'US' in spec.psptype:
                    self.species_order.append(specie)
            else:
                self.species_order.append(specie)

        # then go through again and add species that explicitly include
        # norm-conserving and PAWs
        for specie in self.species:
            spec = self.specdict[specie]
            if self.setups and spec.s in self.setups.keys():  # standalone mod
                if 'US' not in spec.psptype:
                    self.species_order.append(specie)

    def parse_pseudopotential(self, verbose=1):
        """ Parse pseudopotential files to extract information on
        PSP type (US, NC, PAW)
        # of valence states
        ... (can be extended)

        Replaced/Consolidated
         get_nvalence
         get_psptype
        """
        pptype = {}    # extract pseudopotential type
        nel = {}    # extract # of valence electrons
        psppath = self.params_control['psppath']
        for x in self.species:
            el = self.specdict[x].s
            if self.setups and el in self.setups.keys():                    # standalone mod
                # self.psppath+'/'+self.setups[el]
                pspfile = os.path.join(psppath, self.setups[el])
            else:
                # self.psppath+'/'+el+'.UPF'
                pspfile = os.path.join(psppath, el + '.UPF')

            ##########################################
            # extract pseudopotential type (for ordering in the input file)
            ##########################################
            #p = os.popen('egrep \'pseudopotential|pseudo_type\' %s | tr \'"\' \' \'' %(pspfile),'r' )
            p = os.popen(
                'egrep \'pseudopotential|pseudo_type\' %s | tail -n1 | tr \'"\' \' \'' %
                (pspfile), 'r')
            for y in p.readline().split():
                if any([i in y for i in ['NC', 'SL', 'US', 'PAW', '1/r']]):
                    pptype[el] = y
                    break
            p.close()
            self.specdict[x].psptype = pptype[el]
            ##########################################

            ##########################################
            # get total number of electrons in valence
            ##########################################
            p = os.popen(
                r'egrep -i \'z\ valence|z_valence\' %s | tr \'"\' \' \'' %
                (pspfile), 'r')    # standalone mod
            for y in p.readline().split():
                if y[0].isdigit() or y[0] == '.':
                    # *** be careful with rounding in case of partial charges (pseudohydrogen) ***
                    nel[el] = int(round(float(y)))
                    break
            p.close()
            ##########################################
            if verbose:
                print(
                    "Parsed pseudopotential file : %s  (type %s)" %
                    (pspfile, pptype[el]))

        ################################################
        nvalence = np.zeros(len(self.specprops), np.int)
        for i, x in enumerate(self.specprops):
            nvalence[i] = nel[self.specdict[x[0]].s]
        self.nvalence = nvalence
        self.nel = nel
        ################################################

    def write_control(self, f):
        """ Write settings for QE &CONTROL section
        Could use more generalization (but mostly already handled input order).
        """
        print('&CONTROL', file=f)

        keymatch = d_custom_keys['control']

        # strings
        for key in ['calcmode', 'restart', 'psppath', 'prefix', 'outdir',
                    'verbosity', 'title', 'disk_io']:
            val = self.params_control[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s=\'%s\',' % (key, val), file=f)

        # logicals
        for key in ['trism',
                    'constmu',  # 'lcfp',
                    'wf_collect', 'printforces', 'calcstress',
                    'field', 'dipole',
                    ]:
            val = self.params_control[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s= %s,' % (key, bool2str(val)), file=f)

        # integers
        for key in [
            'timestep',
            'ndr',
            'ndw',
            'nstep',
            'iprint',
            'isave',
                'max_seconds']:
            val = self.params_control[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s = %d,' % (key, val), file=f)

        # floats
        # DEBUG: CONVERGENCE
        if self.params_convergence['convergence']:
            print("BE CAREFUL WITH PWSCF VS CP FOR CONVERGENCE PARAMETERS (Ha/Ry)")
            print(
                '  etot_conv_thr = %s,' %
                (num2str(
                    self.params_convergence['etot_conv_thr'] /
                    rydberg)),
                file=f)
            print(
                '  forc_conv_thr = %s,' %
                (num2str(
                    self.params_convergence['fmax'] /
                    rydberg_over_bohr)),
                file=f)
        print('/', file=f)

    def write_controlv0(self, f):
        """ Write settings for QE &CONTROL section
        """
        print('&CONTROL', file=f)
        print(
            '  calculation=\'%s\',' %
            (self.params_control['calcmode']),
            file=f)

        if self.params_control['restart'] == 'from_scratch':
            print('  restart_mode=\'from_scratch\',', file=f)
        else:
            print('  restart_mode=\'restart\',', file=f)

        if self.params_control['verbose'] != 'low':
            print(
                '  verbosity=\'%s\',' %
                (self.params_control['verbosity']),
                file=f)

        print(
            '  pseudo_dir=\'%s\',' %
            (self.params_control['psppath']),
            file=f)

        if self.params_control['timestep'] is not None:
            print('  dt = %d,' % (self.params_control['timestep']), file=f)

        for key in ['ndr', 'ndw', 'nstep', 'iprint', 'isave', 'max_seconds']:
            val = self.params_control[key]
            if val is not None:
                print('  %s = %d,' % (key, val), file=f)

        for key in ['outdir', 'prefix', 'title']:
            val = self.params_control[key]
            if val is not None:
                print('  %s=\'%s\',' % (key, val), file=f)

        if self.params_control['calcstress']:
            print('  tstress=.true.,', file=f)

        #######################################################################
        # electric field / dipole
        #######################################################################
        efield = (self.params_control['field']['status'])
        dipfield = (self.params_control['dipole']['status'])
        if efield or dipfield:
            print('  tefield=.true.,', file=f)
            if dipfield:
                print('  dipfield=.true.,', file=f)
        #######################################################################

# DEBUG: OUTPUT FLAGS
        keymatch = {'printforces': 'tprnfor',
                    'wf_collect': 'wf_collect',
                    'disk_io': 'disk_io',
                    }
        for key in ['printforces', 'wf_collect']:
            val = self.params_control[key]
            if val:
                print('  %s = .true.,' % (keymatch[key]), file=f)

        if self.params_control['disk_io'] in ['high', 'low', 'none']:
            print(
                '  disk_io = \'%s\',' %
                (self.params_control['disk_io']),
                file=f)

# DEBUG: CONVERGENCE
        if self.params_convergence['convergence']:
            print("BE CAREFUL WITH PWSCF VS CP FOR CONVERGENCE PARAMETERS (Ha/Ry)")
            print(
                '  etot_conv_thr = %s,' %
                (num2str(
                    self.params_convergence['etot_conv_thr'] /
                    rydberg)),
                file=f)
            print(
                '  forc_conv_thr = %s,' %
                (num2str(
                    self.params_convergence['fmax'] /
                    rydberg_over_bohr)),
                file=f)

    def write_celldm(self, f):
        """ Format the celldm correctly and write to file
        """
        print(" celldm(i) not implemented!")
        # print('  celldm(1)=1.8897261245650618d0,'   # standalone mod,, file=f)
        # celldm can act as a scaling factor for ibrav=0

    def write_systemv2(self, f):
        """ Write &SYSTEM component of QE input file. Work in progress to generalize the input writing """
        print('&SYSTEM', file=f)

        dref = d_input_types['system']
        keymatch = d_custom_keys['system']

        # strings
        for key in dref['str']:
            val = self.params_system[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s=\'%s\',' % (key, val), file=f)

        # bool
        for key in dref['bool']:
            val = self.params_system[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s= %s,' % (key, bool2str(val)), file=f)

        # int
        for key in dref['int']:
            val = self.params_system[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s=%d,' % (key, val), file=f)

        # floats
        for key in dref['float']:
            val = self.params_system[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s= %s,' % (key, num2str(val)), file=f)
        print('/', file=f)

    def write_system(self, f):
        """ Write &SYSTEM component of QE input file. """
        print('&SYSTEM', file=f)

        # DFT - XCF
        if self.params_system['xc']:
            print('  input_dft = \'%s\',' % (self.params_system['xc']), file=f)

        for key in ['ibrav', 'nat', 'ntyp']:
            val = self.params_system[key]
            if val is not None:
                print('  %s = %d,' % (key, val), file=f)
        if self.params_system['celldm'] and self.params_system['ibrav'] != 0:
            self.write_celldm(f)

        print(
            '  ecutwfc = %.3f,' %
            (self.params_system['pw'] / rydberg), file=f)
        print(
            '  ecutrho = %.3f,' %
            (self.params_system['dw'] / rydberg), file=f)
        if self.params_system['fw'] is not None:
            print(
                '  ecutfock = %.3f,' %
                (self.params_system['fw'] / rydberg), file=f)

        # number of bands
        nbands = self.params_system['nbands']
        if nbands:
            if nbands > 0:
                totband = int(self.nbands)
            else:  # if self.nbands is negative create -self.nbands extra bands
                totband = int(np.sum(self.nvalence) - nbands)
            print('  nbnd = %s,' % (str(totband)), file=f)

        # handling spin polarization
        spinpol = self.params_system['spinpol']
        if spinpol:
            print('  nspin = 2,', file=f)
            spcount = 1
            for species in self.species:    # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print('  starting_magnetization(%d)=%s,' %
                      (spcount, num2str(float(mag))), file=f)
                spcount += 1
        elif self.params_system['noncollinear']:
            print('  noncolin=.true.,', file=f)
            if self.params_system['spinorbit']:
                print('  lspinorb=.true.', file=f)
            spcount = 1
            for species in self.species:  # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print('  starting_magnetization(%d)=%s,' %
                      (spcount, num2str(float(mag))), file=f)
                spcount += 1

        # handling magnetic polarization
        if self.params_control['calcmode'] != 'hund':
            inimagscale = 1.0
        else:
            inimagscale = 0.9

        if self.params_system['fix_magmom']:
            assert spinpol
            self.totmag = self.summed_magmoms
            print('  tot_magnetization = %s,' %
                  (num2str(self.totmag * inimagscale)), file=f)
        elif self.params_system['tot_magnetization'] != -1:
            if self.params_system['tot_magnetization'] == 'hund':
                # DEBUG: may break if ever used
                from atomic_configs import hundmag
                self.totmag = sum([hundmag(x)
                                   for x in self.atoms.get_chemical_symbols()])
            else:
                self.totmag = self.params_system['tot_magnetization']
            print('  tot_magnetization = %s,' %
                  (num2str(self.totmag * inimagscale)), file=f)

        # smearing
        if 'cp' not in self.params_control['calcmode']:
            occupations = self.params_system['occupations']
            smearing = self.params_system['smearing']
            sigma = self.params_system['sigma']

            print('  occupations = \'%s\',' % (occupations), file=f)
            if abs(sigma) > 1e-13 and occupations != 'tetrahedra':
                print('  smearing = \'%s\',' % (smearing), file=f)
                #print('  degauss = %s,' %( num2str(sigma/rydberg)), file=f)
                print('  degauss = %.5f,' % (sigma / rydberg), file=f)
            else:
                if spinpol:
                    assert self.params_system['fix_magmom']
                print(
                    " *** ASSUMING INSULATOR -> SMEARING = %.3f eV (OCCUPATIONS = %s ) ***" %
                    (sigma, occupations))

        # ESM/RISM/ molecule-dipole
        if self.params_system['isolated'] is not None:
            print('  assume_isolated = \'%s\',' %
                  (self.params_system['isolated']), file=f)
        if self.params_system['esm_bc'] is not None:
            print(
                '  esm_bc = \'%s\',' %
                (self.params_system['esm_bc']),
                file=f)

        # charge
        for key in ['tot_charge']:
            val = self.params_system[key]
            if val is not None:
                print('  %s = %d,' % (key, val), file=f)

        if self.params_system['starting_charge'] is not None:
            for i, s in enumerate(self.species):
                spec = self.specdict[s]
                el = spec.s
                qstart = spec.qstart
                print(
                    '  starting_charge(%d)= %s,' %
                    (i + 1, num2str(qstart)), file=f)

        # WHAT IS DIFFERENCE BETWEEN lcpm and lcpmopt?
#         if self.params_control['lcpm']:
#             #print('/\n&FCP', file=f)
#             for key in ['fcp_mu']:
#                 val = self.params_system[key]
#                 if val is not None:
# print('  %s = %d,' %(keymatch[key], val)                        },
# file=f)

        # BEEF
        if self.params_system['beefensemble']:
            # if self.params_system['ensemble_energies']:
            print('  ensemble_energies=.true.,', file=f)
            if self.params_system['printensemble']:
                # if self.params_system['print_ensemble_energies']:
                print('  print_ensemble_energies=.true.,', file=f)
            else:
                print('  print_ensemble_energies=.false.,', file=f)

        #######################################################################
        # electric field / dipole
        #######################################################################
        # first identify if fields are to be included
        include_dipole = 0
        for i in ['tefield', 'dipfield', 'field', 'dipole']:
            if self.params_control[i] is not None:
                include_dipole += 1

        # if dipoles or fields are to be included, define relevant parameters
        keymatch = d_custom_keys['system']
        # if self.params_control['tefield'] or self.params_control['dipfield']:
        if include_dipole > 0:
            # define emaxpos based on vacuum if it is not defined
            #             if self.params_system['emaxpos'] is None:
            #                 self.params_system['emaxpos'] = find_max_empty_space(self.atoms, self.params_system['edir'])

            for key in ['edir', 'eamp', 'emaxpos', 'eopreg']:
                val = self.params_system[key]
                if key in keymatch:
                    key = keymatch[key]
                if val is not None:
                    print("assigning %s" % (key))
                    print('  %s= %s,' % (key, num2str(val)), file=f)

        # DEBUG electric field / dipole
#         edir     = 3
#         efield   = (self.params_control['field']['status']==True)
#         dipfield = (self.params_control['dipole']['status']==True)
#         if dipfield:
#             try:
#                 edir = self.params_control['dipole']['edir']
#             except:
#                 pass
#         elif efield:
#             try:
#                 edir = self.params_control['field']['edir']
#             except:
#                 pass
#         if dipfield or efield:
#             print('  edir=%s,' %( str(edir) ), file=f)
#         if dipfield:
#             if self.params_control['dipole'].has_key('emaxpos'):
#                 emaxpos = self.params_control['dipole']['emaxpos']
#             else:
#                 emaxpos = find_max_empty_space(self.atoms,edir)
#             if self.params_control['dipole'].has_key('eopreg'):
#                 eopreg = self.params_control['dipole']['eopreg']
#             else:
#                 eopreg = 0.025
#             if self.params_control['dipole'].has_key('eamp'):
#                 eamp = self.params_control['dipole']['eamp']
#             else:
#                 eamp = 0.0
#             print('  emaxpos=%s,' %(num2str(emaxpos)), file=f)
#             print('  eopreg=%s,' %(num2str(eopreg)), file=f)
#             print('  eamp=%s,' %(num2str(eamp)), file=f)
#         if efield:
#             if self.params_control['field'].has_key('emaxpos'):
#                 emaxpos = self.params_control['field']['emaxpos']
#             else:
#                 emaxpos = 0.0
#             if self.params_control['field'].has_key('eopreg'):
#                 eopreg = self.params_control['field']['eopreg']
#             else:
#                 eopreg = 0.0
#             if self.params_control['field'].has_key('eamp'):
#                 eamp = self.params_control['field']['eamp']
#             else:
#                 eamp = 0.0
#             print('  emaxpos=%s,' %(num2str(emaxpos)), file=f)
#             print('  eopreg=%s,' %(num2str(eopreg)), file=f)
#             print('  eamp=%s,' %(num2str(eamp)), file=f)

        # higher-level DFT (DFT+U / hybrids)
        if self.params_system['U'] is not None or self.params_system['J'] is not None or self.params_system['U_alpha'] is not None:
            print('  lda_plus_u = .true.,', file=f)
            if self.params_system['J'] is not None:
                print('  lda_plus_u_kind = 1,', file=f)
            else:
                print('  lda_plus_u_kind = 0,', file=f)
            print('  U_projection_type = \'%s\',' %
                  self.params_system['U_projection_type'], file=f)
            if self.params_system['U'] is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    Ui = spec.U
                    print(
                        '  Hubbard_U(%d)= %s,' %
                        (i + 1, num2str(Ui)), file=f)
            if self.params_system['J'] is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    Ji = spec.J
                    print(
                        '  Hubbard_J(%d)= %s,' %
                        (i + 1, num2str(Ji)), file=f)
            if self.params_system['U_alpha'] is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    U_alphai = spec.U_alpha
                    print(
                        '  Hubbard_alpha(%d)= %s,' %
                        (i + 1, num2str(U_alphai)), file=f)

        for key in ['nqx1', 'nqx2', 'nqx3']:
            val = self.params_system[key]
            if val is not None:
                print('  %s = %d,' % (key, val), file=f)

        for key in ['exx_fraction', 'screening_parameter', 'ecutvcut']:
            val = self.params_system[key]
            if val is not None:
                print('  %s = %f,' % (key, val), file=f)

        for key in ['exxdiv_treatment']:
            val = self.params_system[key]
            if val is not None:
                print('  %s = \'%s\',' % (key, val), file=f)

        for key in ['nosym', 'noinv', 'nosym_evc', 'no_t_rev']:
            val = self.params_system[key]
            if val:
                print('  %s = .true.,' % (key), file=f)

        if self.params_system['fft_grid'] is not None:  # RK
            x1, x2, x3 = self.params_system['fft_grid']
            print('  nr1=%d, nr2=%d, nr3=%d,' % (x1, x2, x3), file=f)
        # must be specified in cp is ultrasofts are used
        if self.params_system['usp_grid'] is not None:
            x1, x2, x3 = self.params_system['usp_grid']
            print('  nr1b=%d, nr2b=%d, nr3b=%d,' % (x1, x2, x3), file=f)
        print('/', file=f)

    def write_electrons(self, f):
        """ Write &ELECTRONS component of QE input file. """
        print('&ELECTRONS', file=f)

        dref = d_input_types['electrons']
        keymatch = d_custom_keys['electrons']

        # strings
        keylist = sorted(dref['str'])
        for key in keylist:
            val = self.params_electron[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s=\'%s\',' % (key, val), file=f)

        # bool
        keylist = dref['bool']
        keylist.sort()
        for key in keylist:
            val = self.params_electron[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s= %s,' % (key, bool2str(val)), file=f)

        # int
        keylist = dref['int']
        keylist.sort()
        for key in keylist:
            val = self.params_electron[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s=%d,' % (key, val), file=f)

        # floats
        keylist = dref['float']
        keylist.sort()
        for key in keylist:
            val = self.params_electron[key]
            if key in keymatch:
                key = keymatch[key]
            if val is not None:
                print('  %s= %s,' % (key, num2str(val)), file=f)
        print('/', file=f)

    def write_ions(self, f):
        """ Write &IONS component of QE input file.
        ion_dynamics only specified for certain calculation types and have unique defaults.
        """
        ionssec = self.params_control['calcmode'] in (
            'relax', 'md', 'vc-relax', 'vc-md', 'cp', 'vc_cp', 'cp-wf', 'vc-cp-wf')
        if not ionssec:
            simpleconstr, otherconstr = [], []
        else:
            simpleconstr, otherconstr = convert_constraints(
                self.atoms)  # from qe_utils

        if ionssec:
            print('&IONS', file=f)

#             DEBUG: MAKE ROBUST FOR BORN-OPPENHEIMER IN PWSCF
#             # DISABLE ASE/PW DEFAULT
#             if len(otherconstr)!=0 and 'relax' in self.params_control['calcmode']:    # if constraints
#                 self.params_ion['ion_dynamics'] = 'damp'
#                 self.optdamp = True
#             elif len(otherconstr)!=0 and self.params_control['calcmode']=='md':       # if constraints and BO-MD
#                 self.params_ion['ion_dynamics'] = 'verlet'
#             elif len(otherconstr)==0 and self.ion_dynamics:
#                 # make sure ion_dynamics is compatible with calculation type, otherwise use default
#                 if self.params_ion['ion_dynamics'] in ('langevin','langevin-smc') and self.params_control['calcmode']=='md':
#                     print('  ion_dynamics = \'%s\',' %(self.params_ion['ion_dynamics']), file=f)

            for key in ['ion_dynamics', 'ion_temperature', 'ion_positions']:
                val = self.params_ion[key]
                if val is not None:
                    print('  %s = \'%s\',' % (key, val), file=f)

            # CP ONLY
            if self.params_ion['ion_temperature'] == 'nose' and self.params_control['calcmode'] == 'cp':
                for key in ['nhpcl', 'nhptyp', 'ndega']:
                    val = self.params_ion[key]
                    if val is not None:
                        print('  %s = %d,' % (key, val), file=f)

                if self.params_ion['temperature']:
                    print(
                        '  tempw = %d,' %
                        (self.params_ion['temperature']),
                        file=f)
                if self.params_ion['nhfreq']:
                    # print('  fnosep = %s,' %( formatfreqs(, file=f)
                    # self.params_ion['nhfreq'] ) )  # FIX to correctly order
                    # array for NH freqs
                    fnose_ordered = [self.params_ion['nhfreq'][i.strip(
                        '0123456789')] for i in self.species_order]
                    # only take last 4 entries
                    print('  fnosep = %s,' %
                          (formatfreqs(fnose_ordered[0:4])), file=f)

                # DEBUG: NOT IMPLEMENTED OR CURRENTLY DEFINED
                # if self.nhgrp:
                #    print('  nhgrp = %s,' %(self.nhgrp), file=f)
                # if self.fnhscl:
                #    print('  fnhscl = %s,' %(self.fnhscl), file=f)

            elif self.params_ion['ion_temperature'] == 'rescaling' and self.params_ion['tolp']:
                print('  tolp = %d,' % (self.params_ion['tolp']), file=f)
            print('/', file=f)

    def write_wannier(self, f):
        """ Write &WANNIER component of QE input file.
        """
        if 'wf' in self.params_control['calcmode']:
            print('&WANNIER', file=f)

            keymatch = {'calwf': 'calwf',
                        'nwannsteps': 'nsteps',
                        'nwann_sd_iter': 'nsd',
                        'nwann_tot_iter': 'nit',
                        'wann_tolerance': 'tolw',
                        }

            for key in [
                'calwf',
                'nwann_tot_iter',
                'nwann_sd_iter',
                    'nwannsteps']:
                val = self.params_wannier[key]
                if val is not None:
                    print('  %s = %d,' % (keymatch[key], val), file=f)

            for key in ['wann_tolerance']:
                val = self.params_wannier[key]
                if val is not None:
                    print('  %s = %g,' % (num2str(keymatch[key]), val), file=f)
            print('/', file=f)

    def write_fcp(self, f):
        """ Write &fcp component of QE input file.
        Includes information for setting fermi level as part of RISM
        Only a separate namelist for QE above official 6.1
        """

#        if self.params_control['lfcp'] is not None:
        if self.params_control['constmu'] is not None:
            print("DOING FCP!!!!")
            print('&FCP', file=f)

            for key, val in self.params_fcp.iteritems():
                # print(" FCP %s" %(key))
                if val is not None:
                    print("   setting FCP %s to %s " % (key, val))
                    print('  %s = %s,' % (key, val), file=f)
            print('/', file=f)

    def write_rism(self, f):
        """ Write &RISM component of QE input file.
        Includes information on RISM settings and SOLVENTS info
        """

        if self.params_control['trism'] is not None:
            print('&RISM', file=f)

            keylist = sorted(self.params_rism.keys())

            for key in keylist:
                val = self.params_rism[key]
                if val is not None and key not in [
                    'solvents',
                    'cations',
                    'anions',
                    'solute_epsilons',
                    'solute_sigmas',
                        'solute_lj']:
                    if key in d_input_types['rism']['str']:
                        print('%s' % (key, val), file=f)
                    else:
                        print('  %s = %s,' % (key, val), file=f)

                # SET LJ
                elif key == 'solute_lj':
                    # [self.params_rism['solute_epsilons'] is not None and self.params_rism['solute_sigmas'] is not None:
                    for i, s in enumerate(self.species_order):
                        spec = self.specdict[s]
                        el = spec.s  # noqa: F841
                        LJtype = spec.LJ_type
                        epsi = spec.LJ_epsilon
                        sigi = spec.LJ_sigma

                        if epsi and sigi:
                            LJtype = 'none'
                            print('%s' % (i + 1, LJtype), file=f)
                            print(
                                '  solute_epsilon(%d) = %s !kcal/mol' %
                                (i + 1, num2str(epsi)), file=f)
                            print(
                                '  solute_sigma(%d) = %s !angstrom' %
                                (i + 1, num2str(sigi)), file=f)
                        else:
                            print('%s' % (i + 1, LJtype), file=f)

            print('/', file=f)

    def write_solvents(self, f):
        """ Write Solvents component of QE input file with RISM info.
        """

        if self.params_control['trism'] is not None:
            # SOLVENTS
            print('\nSOLVENTS { mol/L }', file=f)
            # solvent info is included as dictionary { 'type', 'density',
            # 'file'}
            for species in [
                    self.params_rism['solvents'],
                    self.params_rism['cations'],
                    self.params_rism['anions']]:
                if species:
                    for specie in species:
                        print(
                            ' {:<4} {:>4} {:<30}'.format(
                                specie['type'],
                                specie['density'],
                                specie['file']),
                            file=f)
            #print('', file=f)

    def write_cell(self, f):
        """ Write &CELL component of QE input file.
        Includes information on cell, lattice vectors, and atomic positions
        """

        ionssec = self.params_control['calcmode'] not in (
            'scf', 'nscf', 'bands', 'hund')
        if not ionssec:
            simpleconstr, otherconstr = [], []
        else:
            simpleconstr, otherconstr = convert_constraints(
                self.atoms)  # from qe_utils

#         print('/\n&CELL', file=f)

        # cell dynamics only specified for 'vc-relax' or 'vc-md'/'vc-cp' in
        # pw/cp
        if 'vc' in self.params_control['calcmode']:

            print('&CELL', file=f)

            for key in ['cell_dynamics', 'cell_dofree']:
                val = self.params_cell[key]
                if val is not None:
                    print('  %s = \'%s\',' % (key, val), file=f)

            for key in ['press', 'cell_factor']:
                val = self.params_cell[key]
                if val is not None:
                    print('  %s = %s,' % (key, num2str(val)), file=f)

            # DEBUG: CONVERGENCE
            # and self.params_control['calcmode']=='vc-relax':
            if self.params_convergence['pressure'] is not None:
                print >>f, '  press_conv_thr = %s,' % (
                    num2str(self.params_convergence['pressure']))
            print('/', file=f)

        # CELL_PARAMETERS
        print('\nCELL_PARAMETERS { angstrom }', file=f)
        for i in range(3):
            print('%21.15fd0 %21.15fd0 %21.15fd0' %
                  tuple(self.atoms.cell[i]), file=f)
            # do not convert A to bohr

        print('ATOMIC_SPECIES', file=f)

        # self.species_order includes how ultrasofts must precede
        # norm-conserving/PAW in cp). Undefined setups default to USP.
        for specie in self.species_order:
            spec = self.specdict[specie]
            if self.setups and spec.s in self.setups.keys():
                print('.UPF', file=f)
            else:
                # default Dacapo element name USP
                print('.UPF', file=f)

        print('ATOMIC_POSITIONS { %s }' % (self.coordunits), file=f)
        if len(simpleconstr) == 0:
            for species, pos in self.specprops:
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0' %
                    (species, pos[0], pos[1], pos[2]), file=f)
        else:
            for i, (species, pos) in enumerate(self.specprops):
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0   %d  %d  %d' %
                    (species,
                     pos[0],
                        pos[1],
                        pos[2],
                        simpleconstr[i][0],
                        simpleconstr[i][1],
                        simpleconstr[i][2]),
                    file=f)

        if len(otherconstr) != 0:
            print('CONSTRAINTS', file=f)
            if self.constr_tol is None:
                print(len(otherconstr), file=f)
            else:
                print(len(otherconstr), num2str(self.constr_tol), file=f)
            for x in otherconstr:
                print(x, file=f)

    def write_kpts(self, f, overridekpts=None, overridekptshift=None):
        """ Write k-points to QE input file.
        Uses internally defined k-point mesh and shift, but can manually override by entering them explicitly in the function call.
        """
        if overridekpts is None:
            kp = self.params_kpts['kpts']
        else:
            kp = overridekpts
        if overridekptshift is None:
            kpshift = self.params_kpts['kptshift']
        else:
            kpshift = overridekptshift

        if kp == 'gamma':
            print('K_POINTS gamma', file=f)
        else:
            x = np.shape(kp)
            if len(x) == 1:
                print('K_POINTS automatic', file=f)
                print(kp[0], kp[1], kp[2], file=f)
                print(kpshift[0], kpshift[1], kpshift[2], file=f)
            else:
                print('K_POINTS crystal', file=f)
                print(x[0], file=f)
                w = 1. / x[0]
                for k in kp:
                    if len(k) == 3:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' %
                            (k[0], k[1], k[2], w), file=f)
                    else:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' %
                            (k[0], k[1], k[2], k[3]), file=f)

    def writeinputfile(
            self,
            mode=None,
            overridekpts=None,
            overridekptshift=None):
        """ Define and write all of the parameters for the QE input file.
        Defaults to creating 'pw.in'
        """
        if self.atoms is None:
            raise ValueError('no atoms defined')
        else:
            fname = self.pwinputfile
            f = open(fname, 'w')

            self.write_control(f)
            self.write_system(f)
            self.write_electrons(f)
            self.write_ions(f)
            self.write_wannier(f)
            self.write_fcp(f)   # esm/rsim
            self.write_rism(f)  # esm/rsim
            self.write_cell(f)
            self.write_kpts(f, overridekpts, overridekptshift)
            self.write_solvents(f)

            # closing PWscf input file
            f.close()
            print('PWscf/CP input file %s written' % (fname))

    def initialize(self, atoms, verbose=0):
        """ Create the pw.inp input file for manual submission.
        """
        self.atoms = atoms.copy()

        # Make H heavier by default
        if 'H' in self.atoms.get_chemical_symbols():
            if self.deuterate:
                print("Deuterating hydrogen species! m_H -> m_D")
                self.mass_scale('H', 2.0)
            else:
                print("Hydrogen species: not deteurating (m_H = m_H)")

        # now also creates list of Nose-Hoover freqs
        self.atoms2species(verbose=verbose)
        self.check_spinpol()
        # now gets order of atomic species for ordering frequencies and
        # pseudopotentials
        self.get_species_order()
        self.writeinputfile()

        if verbose and self.params_control['outdir'] not in os.listdir(
                os.getcwd()):
            print(
                "*** Generate %s directory prior to job submission ***" %
                (self.params_control['outdir']))

    def mass_scale(self, species, mass_factor):
        """ Enter an atomic species and a factor to scale the mass by.
        """
        masslist = []
        for i, el in enumerate(self.atoms.get_chemical_symbols()):
            if el == species:
                scale = mass_factor
            else:
                scale = 1.0
            masslist.append(scale * self.atoms.get_masses()[i])
        self.atoms.set_masses(masslist)
        print("Scaled mass of %s by factor %f" % (species, mass_factor))

    def reset_epsilon(self, species, LJ_epsilon):
        """ Enter an atomic species and a LJ epsilon value (in kcal/mol) for RISM.
        """
        self.specdict[species].LJ_epsilon = LJ_epsilon
        print("Resetting RISM LJ epsilon of %s to %f" % (species, LJ_epsilon))

    def reset_sigma(self, species, LJ_sigma):
        """ Enter an atomic species and a LJ epsilon value (in kcal/mol) for RISM.
        """
        self.specdict[species].LJ_sigma = LJ_sigma
        print("Resetting RISM LJ sigma of %s to %f" % (species, LJ_sigma))


def espresso(**kwargs):
    """
    Hacked ase interface for Quantum Espresso pw.x:
    """
    pwscf = cpespresso(code='pw', **kwargs)
#     pwscf.set(timestep=None)
#     pwscf.set(restart='from_scratch')
#     pwscf.set(ndr=None)
#     pwscf.set(ndw=None)
#     pwscf.set(iprint=None)
#     pwscf.set(isave=None)
#     pwscf.set(nhfreqs=None)
#     pwscf.set(nstep=150)
#     pwscf.set(convergence=True)
    return pwscf


def rismespresso(**kwargs):
    """
    Hacked ase interface for Quantum Espresso pw.x:
    """
    pwscf = cpespresso(code='rism', **kwargs)
#     pwscf.set(timestep=None)
#     pwscf.set(timestep=None)
#     pwscf.set(restart='from_scratch')
#     pwscf.set(ndr=None)
#     pwscf.set(ndw=None)
#     pwscf.set(iprint=None)
#     pwscf.set(isave=None)
#     pwscf.set(nhfreqs=None)
#     pwscf.set(nstep=150)
#     pwscf.set(convergence=True)
    return pwscf
