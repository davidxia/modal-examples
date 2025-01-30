from pathlib import Path

import modal
from modal_setup_databases import (
    colabfold_image,
    data_path,
    volume,
    volume_path,
)

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds
HOURS = 60 * MINUTES  # seconds

GiB = 1024 # mebibytes

app_name = "example-colabfold-search"
app = modal.App(app_name)


"""
@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    cpu=32,
    memory=196 * GiB,  # 128GB reccommended for
    timeout=4 * HOURS,
)
def reset():
    import subprocess
    import os

    subprocess.run("rm /vol/data/*_db*", check=True, shell=True)

    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = "1"
    setup_env["MMSEQS_NO_INDEX"] = "1"

    for x in ("uniref30_2302", "colabfold_envdb_202108"):
        command = [
            "mmseqs",
            "tsv2exprofiledb",
            f"/vol/data/{x}",
            f"/vol/data/{x}_db",
        ]
        subprocess.run(command, check=True, env=setup_env)
    volume.commit()
"""

@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    cpu=32,
    memory=196 * GiB,  # 128GB reccommended for
    timeout=4 * HOURS,
)
def run_colabfold_search(
    args : str,
    fasta_content :str,
):
    import shlex
    import subprocess
    import tempfile
    volume.reload()

    fasta_filepath = Path(
        tempfile.NamedTemporaryFile(suffix='.fasta', mode='w', delete=False)
        .name
    )
    fasta_filepath.write_text(fasta_content)

    args = shlex.split(args)

    output_dir = Path("msas")

    subprocess.run(
     ["colabfold_search", fasta_filepath, data_path, output_dir] + args,
     check=True)


@app.local_entrypoint()
def main(
    args: str = None,
    fasta_filepath: str = None,
):

    if fasta_filepath is None:
        args = ""

    if fasta_filepath is None:
        fasta_filepath = here / "input.fasta"
    fasta_content = Path(fasta_filepath).read_text()

    # reset.remote()
    run_colabfold_search.remote(args, fasta_content)
