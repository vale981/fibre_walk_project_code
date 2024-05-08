import argparse
import ringfit.data as data
import click

if __name__ == "main":
    parser = argparse.ArgumentParser(
        description="Fitting the transient response of the fibre loops."
    )
    parser.add_argument(
        "file_path", type=str, help="A required integer positional argument"
    )
