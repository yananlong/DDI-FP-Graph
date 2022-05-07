import argparse
import os.path as osp
from collections import defaultdict
from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils import get_drugs_graphs


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class DDIStructureDataset(InMemoryDataset):
    r"""
    Dataset for molecular structures generated from SMILES

    Parameters
    ----------
        root (str): Root directory where the dataset should be saved.
        name (str): Name of the dataset.
        ddi_file_name (str): Name of the DDI pair file.
        structure_file_name (str): Name of the DBID to PWID conversion file.
        pre_transform,  pre_filter (callable, optional): Functions for pre-transform and
            pre-filter. See docs for torch_geometric.data.Data for more details.
    """

    def __init__(
        self,
        root: str,
        name: str,
        ddi_file_name: str,
        structure_file_name: str,
        use_edge_weight: bool = True,
        use_3d_coordinates: bool = False,
        transform: bool = None,
        pre_transform: bool = None,
    ):
        # Unpack parameters and initialize
        self.root = root
        self.name = name
        self.ddi_file_name = ddi_file_name
        self.structure_file_name = structure_file_name
        self.use_edge_weight = use_edge_weight
        self.use_3d_coordinates = use_3d_coordinates
        self.raw_name = "{}.raw".format(name)
        self.proc_name = "{}.pt".format(name)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths)

    @property
    def raw_file_names(self) -> str:
        return self.raw_name

    @property
    def processed_file_names(self) -> str:
        return self.proc_name

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def smp_dir(self) -> str:
        return osp.join(self.root, "SMP")

    @property
    def raw_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip downloading.
        """

        return osp.join(self.raw_dir, self.raw_file_names)

    @property
    def processed_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip processing.
        """
        return osp.join(self.processed_dir, self.processed_file_names)

    def process(self):
        # Read DDI pairs data and structure data
        _ddi = pd.read_table(osp.join(self.root, self.ddi_file_name))
        


class DDIPathwayDataset(InMemoryDataset):
    r"""
    Dataset for pathway data from SMPDB represented as *single* graphs.

    Parameters
    ----------
        root (str): Root directory where the dataset should be saved.
        name (str): Name of the dataset.
        ddi_file_name (str): Name of the DDI pair file.
        db2pw_file_name (str): Name of the DBID to PWID conversion file.
        pre_transform,  pre_filter (callable, optional): Functions for pre-transform and
            pre-filter. See docs for torch_geometric.data.Data for more details.
    """

    def __init__(
        self,
        root: str,
        name: str,
        ddi_file_name: str,
        db2pw_file_name: str,
        use_edge_weight: bool = True,
        use_edge_compartment: bool = False,
        transform: bool = None,
        pre_transform: bool = None,
    ):
        # Unpack parameters and initialize
        self.root = root
        self.name = name
        self.ddi_file_name = ddi_file_name
        self.db2pw_file_name = db2pw_file_name
        self.use_edge_weight = use_edge_weight
        self.use_edge_compartment = use_edge_compartment
        self.raw_name = "{}.raw".format(name)
        self.proc_name = "{}.pt".format(name)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths)

    @property
    def raw_file_names(self) -> str:
        return self.raw_name

    @property
    def processed_file_names(self) -> str:
        return self.proc_name

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def smp_dir(self) -> str:
        return osp.join(self.root, "SMP")

    @property
    def raw_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip downloading.
        """

        return osp.join(self.raw_dir, self.raw_file_names)

    @property
    def processed_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip processing.
        """
        return osp.join(self.processed_dir, self.processed_file_names)

    @staticmethod
    def make_data_list(
        ddi_graph_ar: np.ndarray, use_edge_weight=True, use_edge_compartment=False
    ) -> List[Data]:
        r"""
        Create from a np.ndarray containing interaction graphs and interaction types a
        list of torch_geometric.data.Data object whose default __inc__() function is not
        overridden. Part of this function is adapted from
        torch_geometric.utils.convert.from_networkx().

        Parameters
        ----------
            ddi_graph_ar (np.ndarray): Array that holds the graphs of
                interacting pairs of drugs.
            use_edge_weight (bool): Use edge weight? [Default: True]
            use_edge_compartment (bool): Use edge comparment information?
                [Default: False]

        Returns
        -------
            (List[Data]): Data list for e.g. DataLoader.
        """

        # Initialize object
        result = []
        for i, ddi in enumerate(ddi_graph_ar):
            # Dictionary to hold data attributes
            data = defaultdict(list)

            # Unpack row
            _G, y = ddi
            x = list(_G.nodes(data=False))

            # N.B.: edge indices must have entries with values in [0, num_nodes - 1]
            G = nx.convert_node_labels_to_integers(_G)

            # Edge attributes: weight and compartment
            if G.number_of_edges() > 0:
                edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
            else:
                edge_attrs = {}
            if use_edge_weight or use_edge_compartment:
                for _, _, feat_dict in G.edges(data=True):
                    if set(feat_dict.keys()) != set(edge_attrs):
                        raise ValueError("Not all edges contain the same attributes")
                    for key, val in feat_dict.items():
                        if use_edge_weight:
                            if key == "weight":
                                data["edge_{}".format(key)].append(val)
                        elif use_edge_compartment:
                            if key == "compartment":
                                data["edge_{}".format(key)].append(val)

            # Convert attributes to tensor
            for key, val in data.items():
                try:
                    data[key] = torch.tensor(val)
                except ValueError:
                    print("Error while converting {} to torch.tensor.".format(key))
                    pass

            # Required attributes
            edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
            data["edge_index"] = edge_index.view(2, -1)
            data["x"] = torch.tensor(x).unsqueeze(-1).float()
            data["y"] = torch.tensor(y).unsqueeze(-1)
            PyG_data = Data.from_dict(data)
            PyG_data.num_nodes = G.number_of_nodes()

            result.append(PyG_data)

        return result

    def process(self):
        # Read DDI pairs data and DBID to PWID conversion table
        _ddi = pd.read_table(osp.join(self.root, self.ddi_file_name))
        db2pw = pd.read_table(osp.join(self.root, self.db2pw_file_name))
        ddi = _ddi[
            _ddi["Drug1"].isin(db2pw["DBID"]) & _ddi["Drug2"].isin(db2pw["DBID"])
        ]
        dbids = pd.unique(ddi[["Drug1", "Drug2"]].values.ravel("K"))
        intids = pd.unique(ddi["ID"])
        print(
            "Done reading DDI pairs and biochemical pathway information.",
            "There are {} drugs and {} DDIs pairs of {} types.".format(
                dbids.shape[0], ddi.shape[0], intids.shape[0]
            ),
            flush=True,
        )

        # Convert DBIDs and interaction types to integers
        drug1_ar = (
            ddi["Drug1"].apply(lambda val: np.argwhere(dbids == val).item()).to_numpy()
        )
        drug2_ar = (
            ddi["Drug2"].apply(lambda val: np.argwhere(dbids == val).item()).to_numpy()
        )
        int_ar = (
            ddi["ID"].apply(lambda val: np.argwhere(intids == val).item()).to_numpy()
        )
        ddi_ar = np.column_stack((drug1_ar, drug2_ar, int_ar))

        # Get pathway graphs for each drug
        print("Processing .sbml files...", end="", flush=True)
        G_drugs = list(get_drugs_graphs(dbids, db2pw, self.smp_dir).values())
        print(" Done!", flush=True)

        # Convert species names to integers
        species_ar = np.unique(np.array(nx.compose_all(G_drugs).nodes, dtype=object))

        # np.array of graph objects
        G_drugs_ar = np.array(
            [
                nx.relabel_nodes(G, lambda val: np.argwhere(species_ar == val).item())
                for G in G_drugs
            ],
            dtype=object,
        )

        # Compose the two pathway graphs in the pair
        ddi_graph = []
        for i, (drug1, drug2, int_id) in enumerate(ddi_ar):
            composition = nx.compose_all([G_drugs_ar[drug1], G_drugs_ar[drug2]])
            if composition.number_of_nodes() > 0:
                ddi_graph.append([composition, int_id])
            else:
                print(
                    "Empty composition for pair {} with pathways ({}, {})".format(
                        i, drug1, drug2
                    )
                )
        ddi_graph_ar = np.array(ddi_graph, dtype=object)
        print(
            "Done composing graph objects.",
            "Empty graphs, if any, have been removed.",
            "There are now {} DDI pairs of {} types left.".format(
                ddi_graph_ar.shape[0], np.unique(ddi_graph_ar[:, 1]).shape[0]
            ),
            flush=True,
        )

        # Create data list
        print("Begin creating datalist...", flush=True)
        data_list = self.make_data_list(
            ddi_graph_ar,
            use_edge_weight=self.use_edge_weight,
            use_edge_compartment=self.use_edge_compartment,
        )

        # Collate data
        data, slices = self.collate(data_list)

        # Save data
        print(
            "Saving processed dataset to {}".format(self.processed_paths),
            flush=True,
        )
        torch.save((data, slices), self.processed_paths)


class DDIPathwayStructureDataset(InMemoryDataset):
    r"""
    Dataset for pathway data from SMPDB represented as *single* graphs plus molecular
    structures as *separate* graphs. There are 3 graphs per DDI pair.

    Parameters
    ----------
        root (str): Root directory where the dataset should be saved.
        name (str): Name of the dataset.
        ddi_file_name (str): Name of the DDI pair file.
        db2pw_file_name (str): Name of the DBID to PWID conversion file.
        pre_transform,  pre_filter (callable, optional): Functions for pre-transform and
            pre-filter. See docs for torch_geometric.data.Data for more details.
    """

    def __init__(
        self,
        root: str,
        name: str,
        ddi_file_name: str,
        db2pw_file_name: str,
        use_edge_weight: bool = True,
        use_edge_compartment: bool = False,
        transform: bool = None,
        pre_transform: bool = None,
    ):
        # Unpack parameters and initialize
        self.root = root
        self.name = name
        self.ddi_file_name = ddi_file_name
        self.db2pw_file_name = db2pw_file_name
        self.use_edge_weight = use_edge_weight
        self.use_edge_compartment = use_edge_compartment
        self.raw_name = "{}.raw".format(name)
        self.proc_name = "{}.pt".format(name)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths)

    @property
    def raw_file_names(self) -> str:
        return self.raw_name

    @property
    def processed_file_names(self) -> str:
        return self.proc_name

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def smp_dir(self) -> str:
        return osp.join(self.root, "SMP")

    @property
    def raw_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip downloading.
        """

        return osp.join(self.raw_dir, self.raw_file_names)

    @property
    def processed_paths(self) -> str:
        """
        The absolute filepaths that must be present in order to skip processing.
        """
        return osp.join(self.processed_dir, self.processed_file_names)

    @staticmethod
    def make_data_list(
        ddi_graph_ar: np.ndarray, use_edge_weight=True, use_edge_compartment=False
    ) -> List[Data]:
        r"""
        Create from a np.ndarray containing interaction graphs and interaction types a
        list of torch_geometric.data.Data object whose default __inc__() function is not
        overridden. Part of this function is adapted from
        torch_geometric.utils.convert.from_networkx().

        Parameters
        ----------
            ddi_graph_ar (np.ndarray): Array that holds the graphs of
                interacting pairs of drugs.
            use_edge_weight (bool): Use edge weight? [Default: True]
            use_edge_compartment (bool): Use edge comparment information?
                [Default: False]

        Returns
        -------
            (List[Data]): Data list for e.g. DataLoader.
        """

        # Initialize object
        result = []
        for i, ddi in enumerate(ddi_graph_ar):
            # Dictionary to hold data attributes
            data = defaultdict(list)

            # Unpack row
            _G, y = ddi
            x = list(_G.nodes(data=False))

            # N.B.: edge indices must have entries with values in [0, num_nodes - 1]
            G = nx.convert_node_labels_to_integers(_G)

            # Edge attributes: weight and compartment
            if G.number_of_edges() > 0:
                edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
            else:
                edge_attrs = {}
            if use_edge_weight or use_edge_compartment:
                for _, _, feat_dict in G.edges(data=True):
                    if set(feat_dict.keys()) != set(edge_attrs):
                        raise ValueError("Not all edges contain the same attributes")
                    for key, val in feat_dict.items():
                        if use_edge_weight:
                            if key == "weight":
                                data["edge_{}".format(key)].append(val)
                        elif use_edge_compartment:
                            if key == "compartment":
                                data["edge_{}".format(key)].append(val)

            # Convert attributes to tensor
            for key, val in data.items():
                try:
                    data[key] = torch.tensor(val)
                except ValueError:
                    print("Error while converting {} to torch.tensor.".format(key))
                    pass

            # Required attributes
            edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
            data["edge_index"] = edge_index.view(2, -1)
            data["x"] = torch.tensor(x).unsqueeze(-1).float()
            data["y"] = torch.tensor(y).unsqueeze(-1)
            PyG_data = Data.from_dict(data)
            PyG_data.num_nodes = G.number_of_nodes()

            result.append(PyG_data)

        return result

    def process(self):
        # Read DDI pairs data and DBID to PWID conversion table
        _ddi = pd.read_table(osp.join(self.root, self.ddi_file_name))
        db2pw = pd.read_table(osp.join(self.root, self.db2pw_file_name))
        ddi = _ddi[
            _ddi["Drug1"].isin(db2pw["DBID"]) & _ddi["Drug2"].isin(db2pw["DBID"])
        ]
        dbids = pd.unique(ddi[["Drug1", "Drug2"]].values.ravel("K"))
        intids = pd.unique(ddi["ID"])
        print(
            "Done reading DDI pairs and biochemical pathway information.",
            "There are {} drugs and {} DDIs pairs of {} types.".format(
                dbids.shape[0], ddi.shape[0], intids.shape[0]
            ),
            flush=True,
        )

        # Convert DBIDs and interaction types to integers
        drug1_ar = (
            ddi["Drug1"].apply(lambda val: np.argwhere(dbids == val).item()).to_numpy()
        )
        drug2_ar = (
            ddi["Drug2"].apply(lambda val: np.argwhere(dbids == val).item()).to_numpy()
        )
        int_ar = (
            ddi["ID"].apply(lambda val: np.argwhere(intids == val).item()).to_numpy()
        )
        ddi_ar = np.column_stack((drug1_ar, drug2_ar, int_ar))

        # Get pathway graphs for each drug
        print("Processing .sbml files...", end="", flush=True)
        G_drugs = list(get_drugs_graphs(dbids, db2pw, self.smp_dir).values())
        print(" Done!", flush=True)

        # Convert species names to integers
        species_ar = np.unique(np.array(nx.compose_all(G_drugs).nodes, dtype=object))

        # np.array of graph objects
        G_drugs_ar = np.array(
            [
                nx.relabel_nodes(G, lambda val: np.argwhere(species_ar == val).item())
                for G in G_drugs
            ],
            dtype=object,
        )

        # Compose the two pathway graphs in the pair
        ddi_graph = []
        for i, (drug1, drug2, int_id) in enumerate(ddi_ar):
            composition = nx.compose_all([G_drugs_ar[drug1], G_drugs_ar[drug2]])
            if composition.number_of_nodes() > 0:
                ddi_graph.append([composition, int_id])
            else:
                print(
                    "Empty composition for pair {} with pathways ({}, {})".format(
                        i, drug1, drug2
                    )
                )
        ddi_graph_ar = np.array(ddi_graph, dtype=object)
        print(
            "Done composing graph objects.",
            "Empty graphs, if any, have been removed.",
            "There are now {} DDI pairs of {} types left.".format(
                ddi_graph_ar.shape[0], np.unique(ddi_graph_ar[:, 1]).shape[0]
            ),
            flush=True,
        )

        # Create data list
        print("Begin creating datalist...", flush=True)
        data_list = self.make_data_list(
            ddi_graph_ar,
            use_edge_weight=self.use_edge_weight,
            use_edge_compartment=self.use_edge_compartment,
        )

        # Collate data
        data, slices = self.collate(data_list)

        # Save data
        print(
            "Saving processed dataset to {}".format(self.processed_paths),
            flush=True,
        )
        torch.save((data, slices), self.processed_paths)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-p",
        "--path",
        action="store",
        type=str,
        required=True,
        dest="ROOT",
        help="""The path to the directory containing the DDI information and the folder
                for the SBML files. This will also be where the Dataset is saved.""",
    )
    p.add_argument(
        "-i",
        "--ddi-file",
        action="store",
        type=str,
        default="ddi_pairs.csv",
        dest="DDI_FILE",
        help="Name of DDI pairs file.",
    )
    p.add_argument(
        "-c",
        "--convert-file",
        action="store",
        type=str,
        default="dbid_pwid.csv",
        dest="CONVERT_FILE",
        help="Name of DBID-PWID conversion file.",
    )
    p.add_argument(
        "-w",
        "--weights",
        action="store_true",
        default=False,
        dest="USE_EDGE_WEIGHT",
        help="Include edge weights? [Default: False]",
    )
    p.add_argument(
        "-k",
        "--compartments",
        action="store_true",
        default=False,
        dest="USE_EDGE_COMPARTMENT",
        help="Include edge compartments? [Default: False]",
    )
    p.add_argument(
        "--hetero",
        action="store_true",
        default=False,
        dest="HETERO",
        help="Prepare heterogeneous graph dataset? [Default: False]",
    )

    return p.parse_args()


def main(args):
    # Test
    DrugBank_DDI = DDIPathwayDataset(
        root=args.ROOT,
        name="DDI_SMPDB"
        + ("_weighted" if args.USE_EDGE_WEIGHT else "")
        + ("_compartments" if args.USE_EDGE_COMPARTMENT else ""),
        ddi_file_name=args.DDI_FILE,
        db2pw_file_name=args.CONVERT_FILE,
        use_edge_weight=args.USE_EDGE_WEIGHT,
        use_edge_compartment=args.USE_EDGE_COMPARTMENT,
    )
    n_DDI = DrugBank_DDI.__len__()
    print(DrugBank_DDI)


if __name__ == "__main__":
    main(get_args())
