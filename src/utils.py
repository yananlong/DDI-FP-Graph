import json
import os
from collections.abc import Iterable
from itertools import product
from typing import List, Union

import libsbml
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.drawing.nx_pydot import graphviz_layout


def _dbid_to_pwid(dbid: str, db2pw: pd.DataFrame) -> np.array:
    r"""
    Convert DBID to PWID

    Parameters
    ----------
        dbid (str): DBID
        db2pw (pd.DataFrame): DBID/PWID conversion table

    Returns
    -------
        (np.array): Array of PWID(s)
    """

    return db2pw[db2pw["DBID"] == dbid]["PW ID"].to_numpy()


def _sbml2graph(
    sbml: libsbml.Model, directed: bool = True, reactants_to_products: bool = False
) -> Union[nx.DiGraph, nx.Graph]:
    r"""
    Extract information from an SBML object read from file
    and convert it to a NetworkX (Di)Graph

    Parameters
    ----------
        sbml (libsbml.Model):
            The SBML object, created with libsbml.SBMLReader().readSBMLFromFile()
        directed (bool): Use directed graph? [default: True]

    Returns
    -------
        (Union[nx.DiGraph, nx.Graph]): NetworkX graph object
    """

    G = nx.DiGraph() if directed else nx.Graph()

    for r in sbml.getListOfReactions():
        for reac, prod, modf in product(
            r.getListOfReactants(), r.getListOfProducts(), r.getListOfModifiers()
        ):
            # Reactant to catalyst
            G.add_weighted_edges_from(
                [(reac.getSpecies(), modf.getSpecies(), reac.getStoichiometry())],
                compartment=r.getCompartment(),
            )

            # Catalyst to product
            G.add_weighted_edges_from(
                [(modf.getSpecies(), prod.getSpecies(), prod.getStoichiometry())],
                compartment=r.getCompartment(),
            )

            # Reactant to product
            if reactants_to_products:
                G.add_weighted_edges_from(
                    [(reac.getSpecies(), prod.getSpecies(), 1)],
                    compartment=r.getCompartment(),
                )

    return G


def _get_drug_graph(
    dbid: str, db2pw: pd.DataFrame, SMPDIR: str, **kwargs
) -> Union[nx.DiGraph, nx.Graph]:
    r"""
    Get the molecular pathway graph for the drug with given DBID

    Parameters
    ----------
        dbid (str): The DBID of a given drug
        db2pw (pd.DataFrame): DBID/PWID conversion table
        SMPDIR (str): Path to .sbml files
        **kwargs: Additional parameters for libsbml.readSBMLFromFile()

    Returns
    -------
        (Union[nx.DiGraph, nx.Graph]): NetworkX graph object
    """

    pwids = _dbid_to_pwid(dbid, db2pw)
    G = nx.compose_all(
        [
            _sbml2graph(
                libsbml.readSBMLFromFile(
                    os.path.join(SMPDIR, "{}.sbml".format(pwid)), **kwargs
                ).getModel()
            )
            for pwid in pwids
        ]
    )

    return G


def get_drugs_graphs(dbids: Iterable[str], db2pw: pd.DataFrame, SMPDIR: str) -> dict:
    r"""
    Wrapper around _get_drug_graph to get graph for all drugs

    Parameter
    ---------
        dbid (str): The DBID of a given drug
        db2pw (pd.DataFrame): DBID/PWID conversion table
        SMPDIR (str): Path to folder containing .sbml files

    Returns
    -------
        (dict): Dictionary of graphs as {DBID: graph object}
    """

    return {idx: _get_drug_graph(idx, db2pw, SMPDIR) for idx in dbids}


def get_interaction_graph(
    dbids: Iterable[str], G_drugs: dict
) -> Union[nx.DiGraph, nx.Graph]:
    r"""
    Get the interaction graph for the given list of DBIDs

    Parameters
    ----------
        dbids (dict): Iterable of DBIDs

    Returns
    -------
        (Union[nx.DiGraph, nx.Graph]): Single composed graph for the given DBID(s)
    """

    return nx.compose_all([G_drugs[idx] for idx in dbids])


def plot_graph(G: Union[nx.DiGraph, nx.Graph], prog: str = "neato", **plot_params):
    """
    Thin wrapper around nx.draw and nx.drawing.nx_pydot.graphviz_layout for plotting

    Parameters
    ----------
        G (Union[nx.DiGraph(), nx.Graph()]): NetworkX graph object
        prog (str):
            Which graphviz program to use for generating the layout
            cf. https://graphviz.org/docs/layouts/
        **plot_params:
            Additional parameters passed to nx.draw
            cf. https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

    Returns
    -------
        N/A
    """

    nx.draw(G, graphviz_layout(G, prog=prog), **plot_params)
