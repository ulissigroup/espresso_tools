""" Standardization of pseudopotentials in self-contained dictionaries.

Includes PBE and PBEsol quantum espresso (.UPF) pseudopotentials from
 - GBRV high-throughput Vanderbilt USP          [ https://www.physics.rutgers.edu/gbrv/#Li ]
 - Standard Solid State Pseudopotentials (SSSP) [ http://materialscloud.org/sssp/ ]
 - Various user-defined ones

Defaults to GBRV as USP.
"""
import os
from .custom import hpc_settings


def get_pseudopotential_path():
    """ Get the path of where the standardized pseudopotential directories and files are stored.
    These currently include:
    GBRV_PBE_UPF_v1.5,
    PSP,
    SSSP_acc_PBE,
    SSSP_eff_PBE,
    SSSP_acc_PBESOL,
    SSSP_eff_PBESOL,
    newultrasofts,
    user_defined

    :return: the full path housing these directories.
    """
    settings = hpc_settings()
    path = settings['psp_path']
    return path


def populate_pseudopotentials(database, xcf='PBE'):
    """ Standardize how to define setups and where they are located.

    :param database: Choose 'SSSP_acc', 'SSSP_eff', 'GBRV', 'brandon', 'manual', or 'JACAPO'
    :param xcf: Choose setups based on exchange-correlation functional.
    Defaults to 'PBE' and currently only accepts 'PBE' or 'PBEsol'.
    :return: pspdir, setups for input into the espresso calculator
    """
    pseudopath = get_pseudopotential_path()

    keylist = ['SSSP_acc', 'SSSP_eff', 'GBRV', 'brandon', 'manual', 'JACAPO']

    d_psp = {'PBE': {
        'GBRV': {'pspdir': 'GBRV_PBE_UPF_v1.5', 'setups': setups_PBE_USP_GBRV},
        'SSSP_acc': {'pspdir': 'SSSP_acc_PBE', 'setups': setups_PBE_SSSP_acc},
        'SSSP_eff': {'pspdir': 'SSSP_eff_PBE', 'setups': setups_PBE_SSSP_eff},
        'brandon': {'pspdir': 'user_defined', 'setups': setups_brandon},
        'manual': {'pspdir': 'PSP', 'setups': setups_USP_manual, },
        'JACAPO': {'pspdir': 'newultrasofts', 'setups': setups_jacapo},
    }, 'PBEsol': {
        'GBRV': {'pspdir': 'GBRV_PBESOL_UPF_v1.5', 'setups': setups_PBESOL_USP_GBRV},
        'SSSP_acc': {'pspdir': 'SSSP_acc_PBESOL', 'setups': setups_PBESOL_SSSP_acc},
        'SSSP_eff': {'pspdir': 'SSSP_eff_PBESOL', 'setups': setups_PBESOL_SSSP_eff},
    }
    }

    pspdir = None
    setups = None

    if database in keylist:
        pspdir = os.path.join(pseudopath, d_psp[xcf][database]['pspdir'])
        setups = d_psp[xcf][database]['setups']
    return pspdir, setups


# SSSP all
# SSSP dataset for accuracy
setups_PBE_SSSP_acc = {
    'Ag': 'Ag_pbe_v1.4.uspp.f.upf',
    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.upf',
    'Ar': 'Ar.pbe-n-rrkjus_psl.1.0.0.upf',
    'As': 'As.pbe-n-rrkjus_psl.0.2.upf',
    'Au': 'Au_oncv_pbe-1.0.upf',
    'B': 'B.pbe-n-kjpaw_psl.0.1.upf',
    'Ba': 'Ba_oncv_pbe-1.0.upf',
    'Be': 'Be_oncv_pbe-1.0.upf',
    'Bi': 'Bi.pbe-dn-kjpaw_psl.0.2.2.upf',
    'Br': 'Br_pbe_v1.4.uspp.f.upf',
    'C': 'C_pbe_v1.2.uspp.f.upf',
    'Ca': 'Ca_pbe_v1.uspp.f.upf',
    'Cd': 'Cd.pbe-dn-rrkjus_psl.0.3.1.upf',
    'Ce': 'Ce.gga-pbe-paw-v1.0.upf',
    'Cl': 'Cl.pbe-n-rrkjus_psl.1.0.0.upf',
    'Co': 'Co_pbe_v1.2.uspp.f.upf',
    'Cr': 'Cr_pbe_v1.5.uspp.f.upf',
    'Cs': 'Cs_pbe_v1.uspp.f.upf',
    'Cu': 'Cu_pbe_v1.2.uspp.f.upf',
    'Dy': 'Dy.gga-pbe-paw-v1.0.upf',
    'Er': 'Er.gga-pbe-paw-v1.0.upf',
    'Eu': 'Eu.gga-pbe-paw-v1.0.upf',
    'F': 'F_pbe_v1.4.uspp.f.upf',
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.upf',
    'Ga': 'Ga.pbe-dn-kjpaw_psl.1.0.0.upf',
    'Gd': 'Gd.gga-pbe-paw-v1.0.upf',
    'Ge': 'Ge.pbe-dn-kjpaw_psl.1.0.0.upf',
    'H': 'H.pbe-rrkjus_psl.0.1.upf',
    'He': 'He_oncv_pbe-1.0.upf',
    'Hf': 'Hf.pbe-spdfn-kjpaw_psl.1.0.0.upf',
    'Hg': 'Hg_pbe_v1.uspp.f.upf',
    'Ho': 'Ho.gga-pbe-paw-v1.0.upf',
    'I': 'I_pbe_v1.uspp.f.upf',
    'In': 'In.pbe-dn-rrkjus_psl.0.2.2.upf',
    'Ir': 'Ir_pbe_v1.2.uspp.f.upf',
    'K': 'K.pbe-spn-rrkjus_psl.1.0.0.upf',
    'Kr': 'Kr.pbe-n-rrkjus_psl.0.2.3.upf',
    'La': 'La.gga-pbe-paw-v1.0.upf',
    'Li': 'Li_pbe_v1.4.uspp.f.upf',
    'Lu': 'Lu.gga-pbe-paw-v1.0.upf',
    'Mg': 'Mg_pbe_v1.4.uspp.f.upf',
    'Mn': 'Mn.pbe-spn-kjpaw_psl.0.3.1.upf',
    'Mo': 'Mo_oncv_pbe-1.0.upf',
    'N': 'N.pbe.theos.upf',
    'Na': 'Na_pbe_v1.uspp.f.upf',
    'Nb': 'Nb.pbe-spn-kjpaw_psl.0.3.0.upf',
    'Nd': 'Nd.gga-pbe-paw-v1.0.upf',
    'Ne': 'Ne.pbe-n-kjpaw_psl.1.0.0.upf',
    'Ni': 'Ni_pbe_v1.4.uspp.f.upf',
    'O': 'O.pbe-n-kjpaw_psl.0.1.upf',
    'Os': 'Os.pbe-spfn-rrkjus_psl.1.0.0.upf',
    'P': 'P.pbe-n-rrkjus_psl.1.0.0.upf',
    'Pb': 'Pb.pbe-dn-kjpaw_psl.0.2.2.upf',
    'Pd': 'Pd.pbe-spn-kjpaw_psl.1.0.0.upf',
    'Pm': 'Pm.gga-pbe-paw-v1.0.upf',
    'Po': 'Po.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Pr': 'Pr.gga-pbe-paw-v1.0.upf',
    'Pt': 'Pt.pbe-spfn-rrkjus_psl.1.0.0.upf',
    'Rb': 'Rb_oncv_pbe-1.0.upf',
    'Re': 'Re_pbe_v1.2.uspp.f.upf',
    'Rh': 'Rh.pbe-spn-kjpaw_psl.1.0.0.upf',
    'Rn': 'Rn.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Ru': 'Ru_oncv_pbe-1.0.upf',
    'S': 'S_pbe_v1.2.uspp.f.upf',
    'Sb': 'Sb_pbe_v1.4.uspp.f.upf',
    'Sc': 'Sc_pbe_v1.uspp.f.upf',
    'Se': 'Se_pbe_v1.uspp.f.upf',
    'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.upf',
    'Sm': 'Sm.gga-pbe-paw-v1.0.upf',
    'Sn': 'Sn_pbe_v1.uspp.f.upf',
    'Sr': 'Sr.pbe-spn-rrkjus_psl.1.0.0.upf',
    'Ta': 'Ta.pbe-spfn-rrkjus_psl.1.0.0.upf',
    'Tb': 'Tb.gga-pbe-paw-v1.0.upf',
    'Tc': 'Tc_oncv_pbe-1.0.upf',
    'Te': 'Te_pbe_v1.uspp.f.upf',
    'Ti': 'Ti_pbe_v1.4.uspp.f.upf',
    'Tl': 'Tl.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Tm': 'Tm.gga-pbe-paw-v1.0.upf',
    'V': 'V_pbe_v1.uspp.f.upf',
    'W': 'W_pbe_v1.2.uspp.f.upf',
    'Xe': 'Xe.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Y': 'Y_pbe_v1.uspp.f.upf',
    'Yb': 'Yb.gga-pbe-paw-v1.0.upf',
    'Zn': 'Zn_pbe_v1.uspp.f.upf',
    'Zr': 'Zr_pbe_v1.uspp.f.upf',
}
# SSSP dataset for high-throughput efficiency
setups_PBE_SSSP_eff = {
    'Ag': 'Ag_pbe_v1.4.uspp.f.upf',
    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.upf',
    'Ar': 'Ar.pbe-n-rrkjus_psl.1.0.0.upf',
    'As': 'As.pbe-n-rrkjus_psl.0.2.upf',
    'Au': 'Au_oncv_pbe-1.0.upf',
    'B': 'B.pbe-n-kjpaw_psl.0.1.upf',
    'Ba': 'Ba_oncv_pbe-1.0.upf',
    'Be': 'Be_pbe_v1.4.uspp.f.upf',
    'Bi': 'Bi.pbe-dn-kjpaw_psl.0.2.2.upf',
    'Br': 'Br_pbe_v1.4.uspp.f.upf',
    'C': 'C_pbe_v1.2.uspp.f.upf',
    'Ca': 'Ca_pbe_v1.uspp.f.upf',
    'Cd': 'Cd.pbe-dn-rrkjus_psl.0.3.1.upf',
    'Ce': 'Ce.gga-pbe-paw-v1.0.upf',
    'Cl': 'Cl_pbe_v1.4.uspp.f.upf',
    'Co': 'Co_pbe_v1.2.uspp.f.upf',
    'Cr': 'Cr_pbe_v1.5.uspp.f.upf',
    'Cs': 'Cs_pbe_v1.uspp.f.upf',
    'Cu': 'Cu_pbe_v1.2.uspp.f.upf',
    'Dy': 'Dy.gga-pbe-paw-v1.0.upf',
    'Er': 'Er.gga-pbe-paw-v1.0.upf',
    'Eu': 'Eu.gga-pbe-paw-v1.0.upf',
    'F': 'F_pbe_v1.4.uspp.f.upf',
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.upf',
    'Ga': 'Ga.pbe-dn-rrkjus_psl.0.2.upf',
    'Gd': 'Gd.gga-pbe-paw-v1.0.upf',
    'Ge': 'Ge.pbe-dn-kjpaw_psl.1.0.0.upf',
    'H': 'H.pbe-rrkjus_psl.0.1.upf',
    'He': 'He_oncv_pbe-1.0.upf',
    'Hf': 'Hf.pbe-spn-rrkjus_psl.0.2.upf',
    'Hg': 'Hg_pbe_v1.uspp.f.upf',
    'Ho': 'Ho.gga-pbe-paw-v1.0.upf',
    'I': 'I_pbe_v1.uspp.f.upf',
    'In': 'In.pbe-dn-rrkjus_psl.0.2.2.upf',
    'Ir': 'Ir_pbe_v1.2.uspp.f.upf',
    'K': 'K.pbe-spn-rrkjus_psl.1.0.0.upf',
    'Kr': 'Kr.pbe-n-rrkjus_psl.0.2.3.upf',
    'La': 'La.gga-pbe-paw-v1.0.upf',
    'Li': 'Li_pbe_v1.4.uspp.f.upf',
    'Lu': 'Lu.gga-pbe-paw-v1.0.upf',
    'Mg': 'Mg_pbe_v1.4.uspp.f.upf',
    'Mn': 'Mn.pbe-spn-kjpaw_psl.0.3.1.upf',
    'Mo': 'Mo_oncv_pbe-1.0.upf',
    'N': 'N.pbe.theos.upf',
    'Na': 'Na_pbe_v1.uspp.f.upf',
    'Nb': 'Nb.pbe-spn-kjpaw_psl.0.3.0.upf',
    'Nd': 'Nd.gga-pbe-paw-v1.0.upf',
    'Ne': 'Ne.pbe-n-kjpaw_psl.1.0.0.upf',
    'Ni': 'Ni_pbe_v1.4.uspp.f.upf',
    'O': 'O_pbe_v1.2.uspp.f.upf',
    'Os': 'Os_pbe_v1.2.uspp.f.upf',
    'P': 'P.pbe-n-rrkjus_psl.1.0.0.upf',
    'Pb': 'Pb.pbe-dn-kjpaw_psl.0.2.2.upf',
    'Pd': 'Pd.pbe-spn-kjpaw_psl.1.0.0.upf',
    'Pm': 'Pm.gga-pbe-paw-v1.0.upf',
    'Po': 'Po.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Pr': 'Pr.gga-pbe-paw-v1.0.upf',
    'Pt': 'Pt_pbe_v1.4.uspp.f.upf',
    'Rb': 'Rb_oncv_pbe-1.0.upf',
    'Re': 'Re_pbe_v1.2.uspp.f.upf',
    'Rh': 'Rh.pbe-spn-kjpaw_psl.1.0.0.upf',
    'Rn': 'Rn.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Ru': 'Ru_oncv_pbe-1.0.upf',
    'S': 'S_pbe_v1.2.uspp.f.upf',
    'Sb': 'Sb_pbe_v1.4.uspp.f.upf',
    'Sc': 'Sc_pbe_v1.uspp.f.upf',
    'Se': 'Se_pbe_v1.uspp.f.upf',
    'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.upf',
    'Sm': 'Sm.gga-pbe-paw-v1.0.upf',
    'Sn': 'Sn_pbe_v1.uspp.f.upf',
    'Sr': 'Sr.pbe-spn-rrkjus_psl.1.0.0.upf',
    'Ta': 'Ta_pbe_v1.uspp.f.upf',
    'Tb': 'Tb.gga-pbe-paw-v1.0.upf',
    'Tc': 'Tc_oncv_pbe-1.0.upf',
    'Te': 'Te_pbe_v1.uspp.f.upf',
    'Ti': 'Ti_pbe_v1.4.uspp.f.upf',
    'Tl': 'Tl.pbe-dn-rrkjus_psl.0.2.3.upf',
    'Tm': 'Tm.gga-pbe-paw-v1.0.upf',
    'V': 'V_pbe_v1.uspp.f.upf',
    'W': 'W_pbe_v1.2.uspp.f.upf',
    'Xe': 'Xe.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Y': 'Y_pbe_v1.uspp.f.upf',
    'Yb': 'Yb.gga-pbe-paw-v1.0.upf',
    'Zn': 'Zn_pbe_v1.uspp.f.upf',
    'Zr': 'Zr_pbe_v1.uspp.f.upf',
}
# SSSP dataset for accuracy with PBEsol
setups_PBESOL_SSSP_acc = {
    'Ag': 'ag_pbesol_v1.4.uspp.F.UPF',
    'Al': 'Al.pbesol-n-kjpaw_psl.1.0.0.UPF',
    'Ar': 'Ar.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'As': 'As.pbesol-n-rrkjus_psl.0.2.UPF',
    'Au': 'Au_ONCV_PBEsol-1.0.oncvpsp.upf',
    'B': 'B.pbesol-n-kjpaw_psl.0.1.UPF',
    'Ba': 'Ba_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Be': 'Be_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Bi': 'Bi.pbesol-dn-kjpaw_psl.0.2.2.UPF',
    'Br': 'br_pbesol_v1.4.uspp.F.UPF',
    'C': 'c_pbesol_v1.2.uspp.F.UPF',
    'Ca': 'ca_pbesol_v1.uspp.F.UPF',
    'Cd': 'Cd.pbesol-dn-rrkjus_psl.0.3.1.UPF',
    'Ce': 'Ce.GGA-PBESOL-paw.UPF',
    'Cl': 'Cl.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Co': 'co_pbesol_v1.2.uspp.F.UPF',
    'Cr': 'cr_pbesol_v1.5.uspp.F.UPF',
    'Cs': 'cs_pbesol_v1.uspp.F.UPF',
    'Cu': 'cu_pbesol_v1.2.uspp.F.UPF',
    'Dy': 'Dy.GGA-PBESOL-paw.UPF',
    'Er': 'Er.GGA-PBESOL-paw.UPF',
    'Eu': 'Eu.GGA-PBESOL-paw.UPF',
    'F': 'f_pbesol_v1.4.uspp.F.UPF',
    'Fe': 'Fe.pbesol-spn-kjpaw_psl.0.2.1.UPF',
    'Ga': 'Ga.pbesol-dn-kjpaw_psl.1.0.0.UPF',
    'Gd': 'Gd.GGA-PBESOL-paw.UPF',
    'Ge': 'Ge.pbesol-dn-kjpaw_psl.1.0.0.UPF',
    'H': 'H.pbesol-rrkjus_psl.0.1.UPF',
    'He': 'He_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Hf': 'Hf.pbesol-spdfn-kjpaw_psl.1.0.0.UPF',
    'Hg': 'hg_pbesol_v1.uspp.F.UPF',
    'Ho': 'Ho.GGA-PBESOL-paw.UPF',
    'I': 'i_pbesol_v1.uspp.F.UPF',
    'In': 'In.pbesol-dn-rrkjus_psl.0.2.2.UPF',
    'Ir': 'ir_pbesol_v1.2.uspp.F.UPF',
    'K': 'K.pbesol-spn-rrkjus_psl.1.0.0.UPF',
    'Kr': 'Kr.pbesol-n-rrkjus_psl.0.2.3.UPF',
    'La': 'La.GGA-PBESOL-paw.UPF',
    'Li': 'li_pbesol_v1.4.uspp.F.UPF',
    'Lu': 'Lu.GGA-PBESOL-paw.UPF',
    'Mg': 'mg_pbesol_v1.4.uspp.F.UPF',
    'Mn': 'Mn.pbesol-spn-kjpaw_psl.0.3.1.UPF',
    'Mo': 'Mo_ONCV_PBEsol-1.0.oncvpsp.upf',
    'N': 'N.pbesol-theos.UPF',
    'Na': 'na_pbesol_v1.uspp.F.UPF',
    'Nb': 'Nb.pbesol-spn-kjpaw_psl.0.3.0.UPF',
    'Nd': 'Nd.GGA-PBESOL-paw.UPF',
    'Ne': 'Ne.pbesol-n-kjpaw_psl.1.0.0.UPF',
    'Ni': 'ni_pbesol_v1.4.uspp.F.UPF',
    'O': 'O.pbesol-n-kjpaw_psl.0.1.UPF',
    'Os': 'Os.pbesol-spfn-rrkjus_psl.1.0.0.UPF',
    'P': 'P.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Pb': 'Pb.pbesol-dn-kjpaw_psl.0.2.2.UPF',
    'Pd': 'Pd.pbesol-spn-kjpaw_psl.1.0.0.UPF',
    'Pm': 'Pm.GGA-PBESOL-paw.UPF',
    'Po': 'Po.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Pr': 'Pr.GGA-PBESOL-paw.UPF',
    'Pt': 'Pt.pbesol-spfn-rrkjus_psl.1.0.0.UPF',
    'Rb': 'Rb_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Re': 're_pbesol_v1.2.uspp.F.UPF',
    'Rh': 'Rh.pbesol-spn-kjpaw_psl.1.0.0.UPF',
    'Rn': 'Rn.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Ru': 'Ru_ONCV_PBEsol-1.0.oncvpsp.upf',
    'S': 's_pbesol_v1.2.uspp.F.UPF',
    'Sb': 'sb_pbesol_v1.4.uspp.F.UPF',
    'Sc': 'sc_pbesol_v1.uspp.F.UPF',
    'Se': 'se_pbesol_v1.uspp.F.UPF',
    'Si': 'Si.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Sm': 'Sm.GGA-PBESOL-paw.UPF',
    'Sn': 'sn_pbesol_v1.uspp.F.UPF',
    'Sr': 'Sr.pbesol-spn-rrkjus_psl.1.0.0.UPF',
    'Ta': 'Ta.pbesol-spfn-rrkjus_psl.1.0.0.UPF',
    'Tb': 'Tb.GGA-PBESOL-paw.UPF',
    'Tc': 'Tc_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Te': 'te_pbesol_v1.uspp.F.UPF',
    'Ti': 'ti_pbesol_v1.4.uspp.F.UPF',
    'Tl': 'Tl.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Tm': 'Tm.GGA-PBESOL-paw.UPF',
    'V': 'v_pbesol_v1.uspp.F.UPF',
    'W': 'w_pbesol_v1.2.uspp.F.UPF',
    'Xe': 'Xe.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Y': 'y_pbesol_v1.uspp.F.UPF',
    'Yb': 'Yb.GGA-PBESOL-paw.UPF',
    'Zn': 'zn_pbesol_v1.uspp.F.UPF',
    'Zr': 'zr_pbesol_v1.uspp.F.UPF',
}
# SSSP dataset for high-throughput efficiency with PBEsol
setups_PBESOL_SSSP_eff = {
    'Ag': 'ag_pbesol_v1.4.uspp.F.UPF',
    'Al': 'Al.pbesol-n-kjpaw_psl.1.0.0.UPF',
    'Ar': 'Ar.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'As': 'As.pbesol-n-rrkjus_psl.0.2.UPF',
    'Au': 'Au_ONCV_PBEsol-1.0.oncvpsp.upf',
    'B': 'B.pbesol-n-kjpaw_psl.0.1.UPF',
    'Ba': 'Ba_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Be': 'be_pbesol_v1.4.uspp.F.UPF',
    'Bi': 'Bi.pbesol-dn-kjpaw_psl.0.2.2.UPF',
    'Br': 'br_pbesol_v1.4.uspp.F.UPF',
    'C': 'c_pbesol_v1.2.uspp.F.UPF',
    'Ca': 'ca_pbesol_v1.uspp.F.UPF',
    'Cd': 'Cd.pbesol-dn-rrkjus_psl.0.3.1.UPF',
    'Ce': 'Ce.GGA-PBESOL-paw.UPF',
    'Cl': 'cl_pbesol_v1.4.uspp.F.UPF',
    'Co': 'co_pbesol_v1.2.uspp.F.UPF',
    'Cr': 'cr_pbesol_v1.5.uspp.F.UPF',
    'Cs': 'cs_pbesol_v1.uspp.F.UPF',
    'Cu': 'cu_pbesol_v1.2.uspp.F.UPF',
    'Dy': 'Dy.GGA-PBESOL-paw.UPF',
    'Er': 'Er.GGA-PBESOL-paw.UPF',
    'Eu': 'Eu.GGA-PBESOL-paw.UPF',
    'F': 'f_pbesol_v1.4.uspp.F.UPF',
    'Fe': 'Fe.pbesol-spn-kjpaw_psl.0.2.1.UPF',
    'Ga': 'Ga.pbesol-dn-rrkjus_psl.0.2.UPF',
    'Gd': 'Gd.GGA-PBESOL-paw.UPF',
    'Ge': 'Ge.pbesol-dn-kjpaw_psl.1.0.0.UPF',
    'H': 'H.pbesol-rrkjus_psl.0.1.UPF',
    'He': 'He_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Hf': 'Hf.pbesol-spn-rrkjus_psl.0.2.UPF',
    'Hg': 'hg_pbesol_v1.uspp.F.UPF',
    'Ho': 'Ho.GGA-PBESOL-paw.UPF',
    'I': 'i_pbesol_v1.uspp.F.UPF',
    'In': 'In.pbesol-dn-rrkjus_psl.0.2.2.UPF',
    'Ir': 'ir_pbesol_v1.2.uspp.F.UPF',
    'K': 'K.pbesol-spn-rrkjus_psl.1.0.0.UPF',
    'Kr': 'Kr.pbesol-n-rrkjus_psl.0.2.3.UPF',
    'La': 'La.GGA-PBESOL-paw.UPF',
    'Li': 'li_pbesol_v1.4.uspp.F.UPF',
    'Lu': 'Lu.GGA-PBESOL-paw.UPF',
    'Mg': 'mg_pbesol_v1.4.uspp.F.UPF',
    'Mn': 'Mn.pbesol-spn-kjpaw_psl.0.3.1.UPF',
    'Mo': 'Mo_ONCV_PBEsol-1.0.oncvpsp.upf',
    'N': 'N.pbesol-theos.UPF',
    'Na': 'na_pbesol_v1.uspp.F.UPF',
    'Nb': 'Nb.pbesol-spn-kjpaw_psl.0.3.0.UPF',
    'Nd': 'Nd.GGA-PBESOL-paw.UPF',
    'Ne': 'Ne.pbesol-n-kjpaw_psl.1.0.0.UPF',
    'Ni': 'ni_pbesol_v1.4.uspp.F.UPF',
    'O': 'o_pbesol_v1.2.uspp.F.UPF',
    'Os': 'os_pbesol_v1.2.uspp.F.UPF',
    'P': 'P.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Pb': 'Pb.pbesol-dn-kjpaw_psl.0.2.2.UPF',
    'Pd': 'Pd.pbesol-spn-kjpaw_psl.1.0.0.UPF',
    'Pm': 'Pm.GGA-PBESOL-paw.UPF',
    'Po': 'Po.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Pr': 'Pr.GGA-PBESOL-paw.UPF',
    'Pt': 'pt_pbesol_v1.4.uspp.F.UPF',
    'Rb': 'Rb_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Re': 're_pbesol_v1.2.uspp.F.UPF',
    'Rh': 'Rh.pbesol-spn-kjpaw_psl.1.0.0.UPF',
    'Rn': 'Rn.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Ru': 'Ru_ONCV_PBEsol-1.0.oncvpsp.upf',
    'S': 's_pbesol_v1.2.uspp.F.UPF',
    'Sb': 'sb_pbesol_v1.4.uspp.F.UPF',
    'Sc': 'sc_pbesol_v1.uspp.F.UPF',
    'Se': 'se_pbesol_v1.uspp.F.UPF',
    'Si': 'Si.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Sm': 'Sm.GGA-PBESOL-paw.UPF',
    'Sn': 'sn_pbesol_v1.uspp.F.UPF',
    'Sr': 'Sr.pbesol-spn-rrkjus_psl.1.0.0.UPF',
    'Ta': 'ta_pbesol_v1.uspp.F.UPF',
    'Tb': 'Tb.GGA-PBESOL-paw.UPF',
    'Tc': 'Tc_ONCV_PBEsol-1.0.oncvpsp.upf',
    'Te': 'te_pbesol_v1.uspp.F.UPF',
    'Ti': 'ti_pbesol_v1.4.uspp.F.UPF',
    'Tl': 'Tl.pbesol-dn-rrkjus_psl.0.2.3.UPF',
    'Tm': 'Tm.GGA-PBESOL-paw.UPF',
    'V': 'v_pbesol_v1.uspp.F.UPF',
    'W': 'w_pbesol_v1.2.uspp.F.UPF',
    'Xe': 'Xe.pbesol-dn-rrkjus_psl.1.0.0.UPF',
    'Y': 'y_pbesol_v1.uspp.F.UPF',
    'Yb': 'Yb.GGA-PBESOL-paw.UPF',
    'Zn': 'zn_pbesol_v1.uspp.F.UPF',
    'Zr': 'zr_pbesol_v1.uspp.F.UPF',
}

# GBRV Ultrasofts
setups_PBE_USP_GBRV = {
    'He': 'He_oncv_pbe-1.0.upf',
    'Ne': 'Ne.pbe-n-kjpaw_psl.1.0.0.upf',
    'Ar': 'Ar.pbe-n-rrkjus_psl.1.0.0.upf',
    'Kr': 'Kr.pbe-n-rrkjus_psl.0.2.3.upf',
    'Xe': 'Xe.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Rn': 'Rn.pbe-dn-rrkjus_psl.1.0.0.upf',
    'Ag': 'ag_pbe_v1.4.uspp.F.UPF',
    'Al': 'al_pbe_v1.uspp.F.UPF',
    'As': 'as_pbe_v1.uspp.F.UPF',
    'Au': 'au_pbe_v1.uspp.F.UPF',
    'B': 'b_pbe_v1.4.uspp.F.UPF',
    'Ba': 'ba_pbe_v1.uspp.F.UPF',
    'Be': 'be_pbe_v1.4.uspp.F.UPF',
    'Br': 'br_pbe_v1.4.uspp.F.UPF',
    'C': 'c_pbe_v1.2.uspp.F.UPF',
    'Ca': 'ca_pbe_v1.uspp.F.UPF',
    'Cd': 'cd_pbe_v1.uspp.F.UPF',
    'Cl': 'cl_pbe_v1.4.uspp.F.UPF',
    'Co': 'co_pbe_v1.2.uspp.F.UPF',
    'Cr': 'cr_pbe_v1.5.uspp.F.UPF',
    'Cs': 'cs_pbe_v1.uspp.F.UPF',
    'Cu': 'cu_pbe_v1.2.uspp.F.UPF',
    'F': 'f_pbe_v1.4.uspp.F.UPF',
    'Fe': 'fe_pbe_v1.5.uspp.F.UPF',
    'Ga': 'ga_pbe_v1.4.uspp.F.UPF',
    'Ge': 'ge_pbe_v1.4.uspp.F.UPF',
    'H': 'h_pbe_v1.4.uspp.F.UPF',
    #'Hf': 'hf_pbe_plus4_v1.uspp.F.UPF',
    'Hf': 'hf_pbe_v1.uspp.F.UPF',
    'Hg': 'hg_pbe_v1.uspp.F.UPF',
    'I': 'i_pbe_v1.uspp.F.UPF',
    'In': 'in_pbe_v1.4.uspp.F.UPF',
    'Ir': 'ir_pbe_v1.2.uspp.F.UPF',
    'K': 'k_pbe_v1.4.uspp.F.UPF',
    'La': 'la_pbe_v1.uspp.F.UPF',
    'Li': 'li_pbe_v1.4.uspp.F.UPF',
    'Mg': 'mg_pbe_v1.4.uspp.F.UPF',
    'Mn': 'mn_pbe_v1.5.uspp.F.UPF',
    'Mo': 'mo_pbe_v1.uspp.F.UPF',
    'N': 'n_pbe_v1.2.uspp.F.UPF',
    'Na': 'na_pbe_v1.5.uspp.F.UPF',
    'Nb': 'nb_pbe_v1.uspp.F.UPF',
    'Ni': 'ni_pbe_v1.4.uspp.F.UPF',
    'O': 'o_pbe_v1.2.uspp.F.UPF',
    'Os': 'os_pbe_v1.2.uspp.F.UPF',
    'P': 'p_pbe_v1.5.uspp.F.UPF',
    'Pb': 'pb_pbe_v1.uspp.F.UPF',
    'Pd': 'pd_pbe_v1.4.uspp.F.UPF',
    'Pt': 'pt_pbe_v1.4.uspp.F.UPF',
    'Rb': 'rb_pbe_v1.uspp.F.UPF',
    'Re': 're_pbe_v1.2.uspp.F.UPF',
    'Rh': 'rh_pbe_v1.4.uspp.F.UPF',
    'Ru': 'ru_pbe_v1.2.uspp.F.UPF',
    'S': 's_pbe_v1.4.uspp.F.UPF',
    'Sb': 'sb_pbe_v1.4.uspp.F.UPF',
    'Sc': 'sc_pbe_v1.uspp.F.UPF',
    'Se': 'se_pbe_v1.uspp.F.UPF',
    'Si': 'si_pbe_v1.uspp.F.UPF',
    'Sn': 'sn_pbe_v1.4.uspp.F.UPF',
    'Sr': 'sr_pbe_v1.uspp.F.UPF',
    'Ta': 'ta_pbe_v1.uspp.F.UPF',
    'Tc': 'tc_pbe_v1.uspp.F.UPF',
    'Te': 'te_pbe_v1.uspp.F.UPF',
    'Ti': 'ti_pbe_v1.4.uspp.F.UPF',
    'Tl': 'tl_pbe_v1.2.uspp.F.UPF',
    'V': 'v_pbe_v1.4.uspp.F.UPF',
    'W': 'w_pbe_v1.2.uspp.F.UPF',
    'Y': 'y_pbe_v1.4.uspp.F.UPF',
    'Zn': 'zn_pbe_v1.uspp.F.UPF',
    'Zr': 'zr_pbe_v1.uspp.F.UPF',
}
setups_PBESOL_USP_GBRV = {
    'Ag': 'ag_pbesol_v1.4.uspp.F.UPF',
    'Al': 'al_pbesol_v1.uspp.F.UPF',
    'As': 'as_pbesol_v1.uspp.F.UPF',
    'Au': 'au_pbesol_v1.uspp.F.UPF',
    'B': 'b_pbesol_v1.4.uspp.F.UPF',
    'Ba': 'ba_pbesol_v1.uspp.F.UPF',
    'Be': 'be_pbesol_v1.4.uspp.F.UPF',
    'Br': 'br_pbesol_v1.4.uspp.F.UPF',
    'C': 'c_pbesol_v1.2.uspp.F.UPF',
    'Ca': 'ca_pbesol_v1.uspp.F.UPF',
    'Cd': 'cd_pbesol_v1.uspp.F.UPF',
    'Cl': 'cl_pbesol_v1.4.uspp.F.UPF',
    'Co': 'co_pbesol_v1.2.uspp.F.UPF',
    'Cr': 'cr_pbesol_v1.5.uspp.F.UPF',
    'Cs': 'cs_pbesol_v1.uspp.F.UPF',
    'Cu': 'cu_pbesol_v1.2.uspp.F.UPF',
    'F': 'f_pbesol_v1.4.uspp.F.UPF',
    'Fe': 'fe_pbesol_v1.5.uspp.F.UPF',
    'Ga': 'ga_pbesol_v1.4.uspp.F.UPF',
    'Ge': 'ge_pbesol_v1.4.uspp.F.UPF',
    'H': 'h_pbesol_v1.4.uspp.F.UPF',
    # 'Hf': 'hf_pbesol_plus4_v1.uspp.F.UPF',
    'Hf': 'hf_pbesol_v1.uspp.F.UPF',
    'Hg': 'hg_pbesol_v1.uspp.F.UPF',
    'I': 'i_pbesol_v1.uspp.F.UPF',
    'In': 'in_pbesol_v1.4.uspp.F.UPF',
    'Ir': 'ir_pbesol_v1.2.uspp.F.UPF',
    'K': 'k_pbesol_v1.4.uspp.F.UPF',
    'La': 'la_pbesol_v1.uspp.F.UPF',
    'Li': 'li_pbesol_v1.4.uspp.F.UPF',
    'Mg': 'mg_pbesol_v1.4.uspp.F.UPF',
    'Mn': 'mn_pbesol_v1.5.uspp.F.UPF',
    'Mo': 'mo_pbesol_v1.uspp.F.UPF',
    'N': 'n_pbesol_v1.2.uspp.F.UPF',
    'Na': 'na_pbesol_v1.5.uspp.F.UPF',
    'Nb': 'nb_pbesol_v1.uspp.F.UPF',
    'Ni': 'ni_pbesol_v1.4.uspp.F.UPF',
    'O': 'o_pbesol_v1.2.uspp.F.UPF',
    'Os': 'os_pbesol_v1.2.uspp.F.UPF',
    'P': 'p_pbesol_v1.5.uspp.F.UPF',
    'Pb': 'pb_pbesol_v1.uspp.F.UPF',
    'Pd': 'pd_pbesol_v1.4.uspp.F.UPF',
    'Pt': 'pt_pbesol_v1.4.uspp.F.UPF',
    'Rb': 'rb_pbesol_v1.uspp.F.UPF',
    'Re': 're_pbesol_v1.2.uspp.F.UPF',
    'Rh': 'rh_pbesol_v1.4.uspp.F.UPF',
    'Ru': 'ru_pbesol_v1.2.uspp.F.UPF',
    'S': 's_pbesol_v1.4.uspp.F.UPF',
    'Sb': 'sb_pbesol_v1.4.uspp.F.UPF',
    'Sc': 'sc_pbesol_v1.uspp.F.UPF',
    'Se': 'se_pbesol_v1.uspp.F.UPF',
    'Si': 'si_pbesol_v1.uspp.F.UPF',
    'Sn': 'sn_pbesol_v1.4.uspp.F.UPF',
    'Sr': 'sr_pbesol_v1.uspp.F.UPF',
    'Ta': 'ta_pbesol_v1.uspp.F.UPF',
    'Tc': 'tc_pbesol_v1.uspp.F.UPF',
    'Te': 'te_pbesol_v1.uspp.F.UPF',
    'Ti': 'ti_pbesol_v1.4.uspp.F.UPF',
    'Tl': 'tl_pbesol_v1.2.uspp.F.UPF',
    'V': 'v_pbesol_v1.4.uspp.F.UPF',
    'W': 'w_pbesol_v1.2.uspp.F.UPF',
    'Y': 'y_pbesol_v1.4.uspp.F.UPF',
    'Zn': 'zn_pbesol_v1.uspp.F.UPF',
    'Zr': 'zr_pbesol_v1.uspp.F.UPF',
}

# User-defined from PSP/QE
setups_jacapo = {
    'Ag': 'Ag.UPF',
    'Al': 'Al.UPF',
    'Ar': 'Ar.UPF',
    'As': 'As.UPF',
    'Au': 'Au.UPF',
    'B': 'B.UPF',
    'Ba': 'Ba.UPF',
    'Be': 'Be.UPF',
    'Bi': 'Bi.UPF',
    'Br': 'Br.UPF',
    'C': 'C.UPF',
    'Ca': 'Ca.UPF',
    'Cd': 'Cd.UPF',
    'Cl': 'Cl.UPF',
    'Co': 'Co.UPF',
    'Cr': 'Cr.UPF',
    'Cs': 'Cs.UPF',
    'Cu': 'Cu.UPF',
    'F': 'F.UPF',
    'Fe': 'Fe.UPF',
    'Ga': 'Ga.UPF',
    'Ge': 'Ge.UPF',
    'H': 'H.UPF',
    'Hf': 'Hf.UPF',
    'Hg': 'Hg.UPF',
    'I': 'I.UPF',
    'In': 'In.UPF',
    'Ir': 'Ir.UPF',
    'K': 'K.UPF',
    'Kr': 'Kr.UPF',
    'La': 'La.UPF',
    'Li': 'Li.UPF',
    'Mg': 'Mg.UPF',
    'Mn': 'Mn.UPF',
    'Mo': 'Mo.UPF',
    'N': 'N.UPF',
    'Na': 'Na.UPF',
    'Nb': 'Nb.UPF',
    'Ni': 'Ni.UPF',
    'O': 'O.UPF',
    'Os': 'Os.UPF',
    'P': 'P.UPF',
    'Pb': 'Pb.UPF',
    'Pd': 'Pd.UPF',
    'Pt': 'Pt.UPF',
    'Re': 'Re.UPF',
    'Rh': 'Rh.UPF',
    'Ru': 'Ru.UPF',
    'S': 'S.UPF',
    'Sb': 'Sb.UPF',
    'Sc': 'Sc.UPF',
    'Se': 'Se.UPF',
    'Si': 'Si.UPF',
    'Sn': 'Sn.UPF',
    'Sr': 'Sr.UPF',
    'Ta': 'Ta.UPF',
    'Te': 'Te.UPF',
    'Ti': 'Ti.UPF',
    'Tl': 'Tl.UPF',
    'V': 'V.UPF',
    'W': 'W.UPF',
    'Xe': 'Xe.UPF',
    'Y': 'Y.UPF',
    'Zn': 'Zn.UPF',
    'Zr': 'Zr.UPF',
}

setups_brandon = {
    'Cu': 'Cu.pbe-d-rrkjus.UPF',
    'I': 'I.pbe-mt_bw.UPF',
    'Ag': 'Ag.pbe-d-rrkjus.UPF',
    'Br': 'Br.pbe-van_mit.UPF',
    'F': 'F.pbe-n-rrkjus_psl.0.1.UPF',
    'H': 'H.pbe-rrkjus.UPF',
    'N': 'N.pbe-rrkjus.UPF',
    'Na': 'Na.pbe-n-mt_bw.UPF',
    'In': 'In.pbe-d-rrkjus.UPF',
    'Li': 'Li.pbe-s-van_ak.UPF',
    'O': 'O.pbe-rrkjus.UPF',
    'Sn': 'Sn.pbe-dn-rrkjus_psl.0.2.UPF',
    'Ti': 'Ti.pbe-sp-van_ak.UPF',
}

setups_USP_manual = {
    'Ag': 'Ag.pbe-dn-rrkjus_psl.0.1.UPF',
    'As': 'As.pbe-n-rrkjus_psl.0.2.UPF',
    'B': 'B.pbe-n-rrkjus_psl.0.1.UPF',
    'Br': 'Br.pbe-n-rrkjus_psl.0.2.UPF',
    'C': 'C.pbe-n-rrkjus_psl.0.1.UPF',
    'Cd': 'Cd.pbe-dn-rrkjus_psl.0.2.UPF',
    'Cl': 'Cl.pbe-n-rrkjus_psl.0.1.UPF',
    'Cu': 'Cu.pbe-dn-rrkjus_psl.0.2.UPF',
    'F': 'F.pbe-n-rrkjus_psl.0.1.UPF',
    'Fe': 'Fe.pbe-spn-rrkjus_psl.0.2.1.UPF',
    'Ge': 'Ge.pbe-dn-rrkjus_psl.0.2.2.UPF',
    'H': 'H.pbe-rrkjus_psl.0.1.UPF',
    'I': 'I.pbe-n-rrkjus_psl.0.2.UPF',
    'In': 'In.pbe-dn-rrkjus_psl.0.2.2.UPF',
    'Ir': 'Ir.pbe-n-rrkjus_psl.0.2.3.UPF',
    'K': 'K.pbe-n-mt.UPF',
    'Li': 'Li.pbe-s-rrkjus_psl.0.2.1.UPF',
    'Mo': 'Mo.pbe-spn-rrkjus_psl.0.2.UPF',
    'N': 'N.pbe-n-rrkjus_psl.0.1.UPF',
    'Na': 'Na.pbe-spn-rrkjus_psl.0.2.UPF',
    'Ni': 'Ni.pbe-n-rrkjus_psl.0.1.UPF',
    'O': 'O.pbe-n-rrkjus_psl.0.1.UPF',
    'P': 'P.pbe-n-rrkjus_psl.0.1.UPF',
    'Pb': 'Pb.pbe-dn-rrkjus_psl.0.2.2.UPF',
    'Pd': 'Pd.pbe-n-rrkjus_psl.0.2.2.UPF',
    'Pt': 'Pt.pbe-n-rrkjus_psl.0.1.UPF',
    'S': 'S.pbe-n-rrkjus_psl.0.1.UPF',
    'Se': 'Se.pbe-n-rrkjus_psl.0.2.UPF',
    'Si': 'Si.pbe-n-rrkjus_psl.0.1.UPF',
    'Sn': 'Sn.pbe-dn-rrkjus_psl.0.2.UPF',
    'Ta': 'Ta.pbe-spn-rrkjus_psl.0.2.UPF',
    'Te': 'Te.pbe-dn-rrkjus_psl.0.2.2.UPF',
    # 'Mg' :  'Mg.pbe-sp-hgh.UPF', # NC, not ultrasoft
    # 'Mg' : 'Mg.pbe-nsp-bpaw.UPF',
    'Mg': 'Mg.pbe-mt_fhi.UPF',  # SL
    'Rb': 'Rb.pbe-sp-hgh.UPF',  # NC, not ultrasoft
}  # previously setups_USP

setups_PAW = {
    'B': 'B.pbe-n-kjpaw_psl.0.1.UPF',
    'H': 'H.pbe-kjpaw_psl.0.1.UPF',
    'Na': 'Na.pbe-spn-kjpaw_psl.0.2.UPF',
    'Mg': 'Mg.pbe-nsp-bpaw.UPF',
}


# DEFAULTS
setups_PBE_USP = setups_PBE_USP_GBRV
setups_PBE_acc = setups_PBE_SSSP_acc
setups_PBE_eff = setups_PBE_SSSP_eff

setups_PBESOL_USP = setups_PBESOL_USP_GBRV
setups_PBESOL_acc = setups_PBESOL_SSSP_acc
setups_PBESOL_eff = setups_PBESOL_SSSP_eff
