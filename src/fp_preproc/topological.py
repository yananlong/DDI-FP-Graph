import json
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load file
with open("/home/yananlong/DrugBank/dbid_smiles.json", mode="r") as f:
    # open("/home/yananlong/DDI/Data/db_smiles.pkl", mode="rb")
    smiles_dictk = json.load(fp=f)

# Helper function
def get_topological(SMILES, params):
    mol = Chem.MolFromSmiles(SMILES)
    fp_ar = np.asarray(AllChem.RDKFingerprint(mol, **params))

    return fp_ar


# Get fingerprints
if __name__ == "__main__":
    params = {"minPath": 1, "maxPath": 7, "fpSize": 2048}
    out_dict = {}
    for i, (dbid, smiles) in enumerate(smiles_dictk.items()):
        try:
            fp_ar = get_topological(smiles, params)
        except Exception as e1:
            print("Error:", dbid, smiles)
            continue
            
        out_dict[dbid] = fp_ar

        if i % 200 == 0:
            print("Molecule {}".format(i), flush=True)

    with open("/home/yananlong/DDI/Data/topological_dict_drugbank.pkl", mode="wb") as f:
        pickle.dump(out_dict, f)
