import json
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

# Load file
with open("/home/yananlong/DDI/Data/db_smiles.pkl", mode="rb") as f:
    smiles_dict_pubchem = pickle.load(file=f)
    
# Helper function
factory_params = {"minPointCount": 2, "maxPointCount": 3, "trianglePruneBins": False}
pharm_bins = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]

def _get_feature_factory(factory_params, pharm_bins):
    # Feature factory
    pharm_factory = SigFactory(
        featFactory=ChemicalFeatures.BuildFeatureFactory(
            osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        ),
        **factory_params
    )
    pharm_factory.SetBins(pharm_bins)
    pharm_factory.Init()
    print(
        "Using 2D pharmacophores with a total of {} bits".format(
            pharm_factory.GetSigSize()
        ),
        flush=True,
    )
    
    # Bit descriptions
    bit_desc = []
    for idx in np.arange(pharm_factory.GetSigSize()):
        bit_desc.append(pharm_factory.GetBitDescription(bitIdx=idx))

    return pharm_factory, bit_desc

def _get_pharmacophore(SMILES, pharm_factory):
    # Molecule object
    mol = Chem.MolFromSmiles(SMILES)

    # Generate fingerprints
    fp = Generate.Gen2DFingerprint(mol=mol, sigFactory=pharm_factory)
    fp_ar = np.asarray(fp.ToList())

    return fp, fp_ar

# Factory
print("Begin building feature factory", flush=True)
pharm_factory, bit_desc = _get_feature_factory(
    factory_params=factory_params, pharm_bins=pharm_bins
)
print("Done building feature factory", flush=True)

# Get fingerprints
pharma_dict = {}
for i, (dbid, smiles) in enumerate(smiles_dict_pubchem.items()):
    if i % 10 == 0:
        print("Begin processing molecule {}".format(i), flush=True)
        
    try:
        _, pharma_fp_ar = _get_pharmacophore(smiles, pharm_factory)
    except Exception as e:
        continue 
    pharma_dict[dbid] = pharma_fp_ar

    
with open("/home/yananlong/DDI/Data/pharmacophore_dict.pkl", mode="wb") as f:
    pickle.dump(pharma_dict, f)