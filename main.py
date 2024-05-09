import argparse
import ringfit.data as data
import click

import click


@click.command()
@click.option("--count", default=1, help="Number of greetings.")
@click.option("--name", prompt="Your name", help="The person to greet.")
def main():
    pass


if __name__ == "main":
    main()
