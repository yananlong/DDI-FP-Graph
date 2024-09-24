import json
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures

# Load file
# with open("/home/yananlong/DDI/Data/smiles_dict.pkl", mode="rb") as f1:
#     smiles_dict_pubchem = pickle.load(file=f1)
with open("/home/yananlong/DrugBank/dbid_smiles.json", mode="r") as f2:
    smiles_dict_drugbank = json.load(fp=f2)

# Helper function
def get_morgan(SMILES, params):
    mol = Chem.MolFromSmiles(SMILES)
    fp_ar = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, **params))

    return fp_ar


# Get fingerprints
if __name__ == "__main__":
    radius = 4
    nBits = 2048
    params = {"radius": radius, "nBits": nBits, "useChirality": True}
    out_dict = {}
    for i, (dbid, smiles) in enumerate(smiles_dict_drugbank.items()):
        try:
            fp_ar = get_morgan(smiles, params)
        except Exception as e1:
            continue
        out_dict[dbid] = fp_ar

        if i % 1000 == 0:
            print("Molecule {}".format(i), flush=True)

    with open(
        f"/home/yananlong/DDI/Data/morgan{radius}-{nBits}_dict_drugbank.pkl", mode="wb"
    ) as f:
        pickle.dump(out_dict, f)
