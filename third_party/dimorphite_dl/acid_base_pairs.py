from third_party.molvs.charge import AcidBasePair
from rdkit import Chem

pair_string = """Nitro	[C,c,N,n,O,o:1]-[NX3:2](=[O:3])-[O-1:4]	[C,c,N,n,O,o:1]-[NX3:2](=[O:3])-[O:4]-[H]	3
Sulfate	[SX4:1](=[O:2])(=[O:3])([O:4]-[C,c,N,n:5])-[O-1:6]	[SX4:1](=[O:2])(=[O:3])([O:4]-[C,c,N,n:5])-[OX2:6]-[H]	5
Sulfonate	[SX4:1](=[O:2])(=[O:3])(-[C,c,N,n:4])-[O-1:5]	[SX4:1](=[O:2])(=[O:3])(-[C,c,N,n:4])-[OX2:5]-[H]	4
Thioic_acid	[C,c,N,n:1](=[O:2])-[S-1:3]	[C,c,N,n:1](=[O:2])-[SX2:3]-[H]	2
Sulfinic_acid	[SX3:1](=[O:2])-[O-1:3]	[SX3:1](=[O:2])-[O:3]-[H]	2
Phosphonate	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[C,c,N,n:4])-[O-1:5]	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[C,c,N,n:4])-[OX2:5]-[H]	2
Phosphonate_ester	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[C,c,N,n,F,Cl,Br,I:5])-[O-1:6]	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[C,c,N,n,F,Cl,Br,I:5])-[OX2:6]-[H]	5
Phosphate	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[O+0:4])-[O-1:5]	[PX4:1](=[O:2])(-[OX2:3]-[H])(-[O+0:4])-[OX2:5]-[H]	2
*Imide	[F,Cl,Br,S,s,P,p:1][#6:2][CX3:3](=[O,S:4])-[N-1:5]([CX3:6]=[O,S:7])	[F,Cl,Br,S,s,P,p:1][#6:2][CX3:3](=[O,S:4])-[NX3+0:5]([CX3:6]=[O,S:7])-[H]	4
Phosphate_diester	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[O+0:5]-[C,c,N,n,F,Cl,Br,I:4])-[O-1:6]	[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[O+0:5]-[C,c,N,n,F,Cl,Br,I:4])-[OX2:6]-[H]	6
Phosphinic_acid	[PX4:1](=[O:2])(-[C,c,N,n,F,Cl,Br,I:3])(-[C,c,N,n,F,Cl,Br,I:4])-[O-1:5]	[PX4:1](=[O:2])(-[C,c,N,n,F,Cl,Br,I:3])(-[C,c,N,n,F,Cl,Br,I:4])-[OX2:5]-[H]	4
Carboxyl	[C:1](=[O:2])-[O-1:3]	[C:1](=[O:2])-[O:3]-[H]	2
$N#CCN [N:1]#[C:2]-[C:3][N:4] [N:1]#[C:2]-[C:3][N+1:4] 3
Phenyl_carboxyl	[c,n,o:1]-[C:2](=[O:3])-[O-1:4]	[c,n,o:1]-[C:2](=[O:3])-[O:4]-[H]	3
*Amide_electronegative	[C:1](=[O:2])-[N-1:3](-[Br,Cl,I,F,S,O,N,P:4])	[C:1](=[O:2])-[N:3](-[Br,Cl,I,F,S,O,N,P:4])-[H]	2
O=C-C=C-OH	[O:1]=[C;R:2]-[C;R:3]=[C;R:4]-[O-1:5]	[O:1]=[C;R:2]-[C;R:3]=[C;R:4]-[O:5]-[H]	4
Anilines_primary	[c:1]-[NX3H2+0:2]	[c:1]-[N+1H3:2]	1
Primary_hydroxyl_amine	[C,c:1]-[O:2]-[NH2:3]	[C,c:1]-[O:2]-[N+1H3:3]	2
Anilines_tertiary	[c:1]-[NX3H0+0:2]	[c:1]-[N+1H1:2]	1
Anilines_secondary	[c:1]-[NX3H1+0:2]	[c:1]-[N+1H2:2]	1
Aromatic_nitrogen_unprotonated	[n+0&H0:1]	[n+1&H1:1]	0
Thioic_acid	[C,c,N,n:1](=[S:2])-[O-1:3]	[C,c,N,n:1](=[S:2])-[OX2:3]-[H]	2
$Se_acid [Se:1](=[O:2])-[O-1:3] [Se:1](=[O:2])-[O:3]-[H]	2
*Azide	[N+0:1]=[N+:2]=[N-1:3]	[N+0:1]=[N+:2]=[N+0:3]-[H]	2
Phenyl_Thiol	[c,n:1]-[S-1:2]	[c,n:1]-[SX2:2]-[H]	1
*Ringed_imide1	[O,S:1]=[C;R:2]([$([#8]),$([#7]),$([#16]),$([#6][Cl]),$([#6]F),$([#6][Br]):3])-[N-1;R:4]([C;R:5]=[O,S:6])	[O,S:1]=[C;R:2]([$([#8]),$([#7]),$([#16]),$([#6][Cl]),$([#6]F),$([#6][Br]):3])-[N;R:4]([C;R:5]=[O,S:6])-[H]	3
Phenol	[c,n,o:1]-[O-1:2]	[c,n,o:1]-[O:2]-[H]	1
*Aromatic_nitrogen_protonated	[nX3-1:1]	[nX3:1]-[H]	0
$ring_nitrogen_protonated [NX3H1&R] [NX3+1&R] 0
*Sulfonamide	[SX4:1](=[O:2])(=[O:3])-[N-1:4]	[SX4:1](=[O:2])(=[O:3])-[NX3+0:4]-[H]	3
Amines_primary_secondary_tertiary	[NX3;!$(NC=O)][C] [NX3+1;!$(NC=O)][C] 	 	0
$SCN [NH:1]-[CX3:2]-[SH0:3] [N+1H2:1]-[CX3:2]-[SHO:3] 0
*Ringed_imide2	[O,S:1]=[C;R:2]-[N-1;R:3]([C;R:4]=[O,S:5])	[O,S:1]=[C;R:2]-[N;R:3]([C;R:4]=[O,S:5])-[H]	2
Peroxide1	[O:1]([$(C=O),$(C[Cl]),$(CF),$(C[Br]),$(CC#N):2])-[O-1:3]	[O:1]([$(C=O),$(C[Cl]),$(CF),$(C[Br]),$(CC#N):2])-[O:3]-[H]	2
Vinyl_alcohol	[C:1]=[C:2]-[O-1:3]	[C:1]=[C:2]-[O:3]-[H]	2
Thiol	[C,N:1]-[S-1:2]	[C,N:1]-[SX2:2]-[H]	1
$BOH [B:1]-[O-1:2] [B:1]-[OH] 1
N-hydroxyamide	[C:1](=[O:2])-[N:3]-[O-1:4]	[C:1](=[O:2])-[N:3]-[O:4]-[H]	3
AmidineGuanidine2	[C:1](-[C,c,N,S:2])=[NX2+0:3]	[C:1](-[C,c,N,S:2])=[NH2+1:3]	2
*Imide2	[O,S:1]=[CX3:2]-[N-1:3]([CX3:4]=[O,S:5])-[H]	[O,S:1]=[CX3:2]-[NX3+0:3]([CX3:4]=[O,S:5])-[H]	2
Peroxide2	[C:1]-[O:2]-[O-1:3]	[C:1]-[O:2]-[O:3]-[H]	2
*Amide	[C:1](=[O:2])-[N-1:3]	[C:1](=[O:2])-[N:3]-[H]	2
AmidineGuanidine1	[N:1]-[C:2](-[N:3])=[NX2:4]-[H:5]	[N:1]-[C:2](-[N:3])=[N+1H2:4]	3
*Indole_pyrrole	[c;R:1]1[c;R:2][c;R:3][c;R:4][n-1;R:5]1	[c;R:1]1[c;R:2][c;R:3][c;R:4][n;R:5]1[H]	4
Alcohol	[C:1]-[O-1:2]	[C:1]-[O:2]-[H]	1
$CNO  [C-1:1]-[NX2:2]=[O:3]   [C:1]-[NX2:2]=[O:3] 1
$N#CCN [N:1]#[C-1:2]-[C:3] [N:1]#[C:2]-[C:3] 1"""


class PairWithIonizationCenter(AcidBasePair):
    def __init__(self, name, acid, base, ionization_center):
        super(PairWithIonizationCenter, self).__init__(name, acid, base)
        self.ionization_center = ionization_center


class AcidBaseFinder():
    def __init__(self, pair_with_ionization_center):
        self.acid_base_pair = pair_with_ionization_center

    def find_acid_base(self, mol):
        idx_acid = None
        idx_base = None

        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetFormalCharge() < 0:
                idx_base = idx
            if atom.GetFormalCharge() > 0:
                idx_acid = idx

        if idx_acid is None:
            for pair in self.acid_base_pair:
                occur = mol.GetSubstructMatch(pair.acid)
                if occur:
                    idx_acid = occur[pair.ionization_center]
                    break

        if idx_base is None:
            for pair in reversed(self.acid_base_pair):
                occur = mol.GetSubstructMatch(pair.base)
                if occur:
                    idx_base = occur[pair.ionization_center]
                    if idx_base != idx_acid:
                        return idx_acid, idx_base
                    else:
                        idx_base = None

        return idx_acid, idx_base


def get_pair():
    ACID_BASE_PAIRS = []
    for line in pair_string.split('\n'):
        line = line.strip()
        split = line.split()
        pair = PairWithIonizationCenter(split[0], split[2], split[1], int(split[3]))
        ACID_BASE_PAIRS.append(pair)
    return ACID_BASE_PAIRS
