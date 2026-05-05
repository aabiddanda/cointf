"""CLI for pGermlinePoly Estimation in VCF."""

import logging

import rich_click as click
import numpy as np
from tqdm import tqdm
import sys

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@main.command()
@click.command(
    help="CLI for simulating data under the Houseworth-Stahl model.",
    context_settings=dict(show_default=True),
)
@click.option(
    "--nu",
    required=True,
    type=float,
    help="Interference parameter.",
)
@click.option(
    "--p",
    "-p",
    required=True,
    type=float,
    help="Probabilty of escape.",
)
@click.option(
    "--L",
    "-l",
    required=True,
    type=float,
    help="Genetic Map Length (Morgans).",
)
@click.option(
    "--n",
    "-n",
    required=True,
    default=100,
    type=int,
    help="Genetic Map Length (Morgans).",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="hs_simulated.csv",
    help="Output CSV file (defaults to stdout)",
)
def simulate(
    nu,
    p,
    L,
    n,
    out,
):
    """CLI for simulating data under the Houseworth-Stahl model."""
    pass


@main.command()
@click.command(
    help="CLI for likelihood inference under the Houseworth-Stahl model.",
    context_settings=dict(show_default=True),
)
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Interference parameter.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="-",
    help="Output parameter inference file and likelihood ratio tests.",
)
def infer(
    input,
    out,
):
    """C"""
    pass


if __name__ == "__main__":
    main()
