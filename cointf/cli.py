"""CLI for simulation and likelihood calculation in the Houseworth-Stahl model"""

import logging

from cointf import Stahl, SimStahl
import polars as pl
import rich_click as click

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="cointf", prog_name="cointf")
def main() -> None:
    """
    Command-line interface for the **cointf** model.

    Use `simulate` to simulate data and `infer` to estimate parameters under the model.
    """


@main.command(
    "simulate",
    help="CLI for simulating data under the Houseworth-Stahl model.",
    context_settings=dict(show_default=True),
)
@click.option(
    "--nu",
    required=True,
    default=1.0,
    type=float,
    help="Interference parameter.",
)
@click.option(
    "--p",
    default=0.0,
    required=True,
    type=float,
    help="Probabilty of escape.",
)
@click.option(
    "--length",
    "-l",
    required=True,
    default=1.0,
    type=float,
    help="Genetic Map Length (Morgans).",
)
@click.option(
    "--n",
    required=True,
    default=100,
    type=int,
    help="Number of independent meioses.",
)
@click.option(
    "--seed",
    required=True,
    default=42,
    type=int,
    help="Random seed.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="-",
    help="Output TSV file (defaults to stdout)",
)
def simulate(
    nu,
    p,
    length,
    n,
    seed,
    out,
):
    """CLI for simulating data under the Houseworth-Stahl model."""
    simulator = SimStahl(L=length, n=n)
    xs = simulator.sim_stahl(p=p, nu=nu, seed=seed)
    if out == "-":
        click.echo("ID\tL\tnu\tp\tX")
        for i, x in enumerate(xs):
            click.echo(f"{i}\t{length}\t{nu}\t{p}\t'{str(x.tolist())}'")
    else:
        with open(out, "w+") as outfile:
            outfile.write("ID\tL\tnu\tp\tX\n")
            for i, x in enumerate(xs):
                outfile.write(f"{i}\t{length}\t{nu}\t{p}\t'{str(x.tolist())}'\n")


@main.command(
    "infer",
    help="CLI for likelihood inference under the Houseworth-Stahl model.",
    context_settings=dict(show_default=True),
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input file of meioses for analysis.",
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
    """CLI for inference under the Houseworth-Stahl model."""
    df = pl.read_csv(input, separator="\t")
    stahl = Stahl()
    L = df["L"].unique()[0]
    co_data = [eval(eval(x)) for x in df["X"].to_numpy()]
    ll = stahl.loglik(co_data, L=L, p=0.1, nu=10)
    print(f"XXX,{ll}")


if __name__ == "__main__":
    main()
