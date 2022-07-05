"""Console script for ptychonn."""
import sys
import click
import ptychonn


@click.group(invoke_without_command=True)
@click.option("--version", help="Print version and return.", is_flag=True)
def main(version):
    """Console script for ptychonn."""
    if version:
        click.echo(f"ptychonn {ptychonn.__version__}")
        return 0

    click.echo("Replace this message by putting your code into "
               "ptychonn.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


main.add_command(ptychonn.infer)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
