'''Define the main entry point for the PtychoNN CLI.

The main entry point without subcommands had only one option --version which
prints the CLI version and exits. All functionality is divided by module
into subcommands like infer or train.
'''
import sys
import click
import ptychonn
import ptychonn._infer.__main__


@click.group(invoke_without_command=True)
@click.option('--version', help='Print version and return.', is_flag=True)
def main(version):
    '''Deep learning of ptychographic imaging.

    https://doi.org/10.1063/5.0013065
    '''
    if version:
        click.echo(f'ptychonn {ptychonn.__version__}')
        return 0
    return 0


main.add_command(ptychonn._infer.__main__.infer_cli)
main.add_command(ptychonn._train.__main__.train_cli)

if __name__ == '__main__':
    sys.exit(main())  # pragma: no cover
