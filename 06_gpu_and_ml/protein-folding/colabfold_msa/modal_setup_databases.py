# # Download and run your own MSA server [WIP]

# ```shell
# modal run modal_setup_databases.py
# ```

# Multiple Sequence Alignment (MSA)
# Goals:
# 1) Run a private MSA server using colabfold, specifically be able to run these
# instructions:
# - https://modal.com/internal/lookup/ta-01JHNNA6E9F0GCGAG1GJ0KMAKR
# 2) Plug this MSA server into Chai so that we can predict many sequences quickly, since
# public ones rate limit users (bc of $$$).


import logging as L
import os
from pathlib import Path
from urllib.parse import urljoin

import modal

app_name = "example-colabfold-setup"
app = modal.App(app_name)

L.basicConfig(
    level=L.INFO,
    format="\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)-22s:%(lineno)-3d] %(message)s\033[0m",
    datefmt="%b %d %H:%M:%S",
)

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds
HOURS = 60 * MINUTES  # seconds

GiB = 1024 # mebibytes

mmseqs_infrastructure_config = {
    'cpu' : 64,
    'memory' : (336) * GiB,
}

print("setting up data storage & paths")
volume = modal.Volume.from_name(
    "example-compbio-colab-v3", create_if_missing=True
)
volume_path = Path("/vol")
data_path = volume_path / "data"
s3_bucket_path = Path("/s3")


# ColabFold uses this commit (May 28, 2023) to create the databases and perform searches.
mmseqs_commit_id = "71dd32ec43e3ac4dabf111bbc4b124f1c66a85f1"
colabfold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "cmake", "zlib1g-dev", "wget", "aria2", "rsync")
    .run_commands(
        "git clone https://github.com/soedinglab/MMseqs2",
        f"cd MMseqs2 && git checkout {mmseqs_commit_id}",
        "cd MMseqs2 && mkdir build",
        "cd MMseqs2/build && cmake -DCMAKE_BUILD_TYPE=RELEASE -DHAVE_ZLIB=1 -DCMAKE_INSTALL_PREFIX=. ..",
        "cd MMseqs2/build && make -j4",
        "cd MMseqs2/build && make install ",
        "ln -s /MMseqs2/build/bin/mmseqs /usr/local/bin/mmseqs",
    )
    .pip_install(
        "colabfold[alphafold-minus-jax]==1.5.5",
        "aria2p==0.12.0",
        "tqdm==4.67.1",
    )
    # TODO: Hack for debugging faster, remove later
    .add_local_file(
        Path(__file__).parent / "input.fasta",
        "/input.fasta")
    .add_local_file(
        Path(__file__).parent / "copy_of_search_with_edits.py",
        "/usr/local/lib/python3.11/site-packages/colabfold/mmseqs/search.py"
    )
)

mmcif_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync")
    .pip_install("tqdm==4.67.1")
)

with mmcif_image.imports():
    import shutil

@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    timeout=8 * HOURS, # TODO ??
    **mmseqs_infrastructure_config
)
def run_mmseqs_create_index(db_filepath, mmseqs_force_merge):
    import subprocess
    import tempfile

    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = "1" if mmseqs_force_merge else "0"
    subprocess.run(
        ["mmseqs", "createindex", db_filepath] +
        [tempfile.mkdtemp(), "--remove-tmp-files", "1"],
        check=True,
        env=setup_env
    )

    volume.commit()


@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=4 * GiB,
    timeout=4 * HOURS,
)
def setup_profile_database(
    base_url: str, db_name: str, mmseqs_no_index: bool, mmseqs_force_merge: bool
):
    import subprocess

    filename = db_name + ".tar.gz"
    url = urljoin(base_url, filename)
    download_filepath = data_path / filename

    L.info(f"downloading from {url} to {download_filepath}")
    download_filepath = download_file(url, data_path, filename)

    extraction_filepath = extract_with_progress(download_filepath)

    db_filepath = (
        extraction_filepath.with_stem(extraction_filepath.stem  + "_db")
    )

    L.info(f"converting TSV to MMseqs2 DB: {db_filepath}")
    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = "1" if mmseqs_force_merge else "0"
    subprocess.run(
        ["mmseqs", "tsv2exprofiledb", extraction_filepath, db_filepath],
        check=True,
        env=setup_env
    )
    volume.commit()

    if not mmseqs_no_index:
        run_mmseqs_create_index.remote(db_filepath, mmseqs_force_merge)


@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=4 * GiB,
    timeout=4 * HOURS,
)
def setup_fasta_database(
    base_url: str, pdb_db_name: str, mmseqs_force_merge: bool
):
    import subprocess

    filename = pdb_db_name + ".fasta.gz"
    url = urljoin(base_url, filename)

    L.info(f"downloading from {url} to {data_path}")
    download_filepath = download_file(url, data_path, filename)

    db_filepath = download_filepath.with_suffix("").with_suffix("")
    L.info(f"creating MMseqs2 DB from {download_filepath} to {db_filepath}")
    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = "1" if mmseqs_force_merge else "0"
    subprocess.run(
        ["mmseqs", "createdb", download_filepath, db_filepath],
        check=True,
        env=setup_env
    )
    volume.commit()

    if not mmseqs_no_index:
        run_mmseqs_create_index.remote(db_filepath, mmseqs_force_merge)


@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=4 * GiB,
    timeout=4 * HOURS,
)
def setup_foldseek_database(base_url: str, pdb_db_name: str):
    filename = pdb_db_name + ".tar.gz"
    url = urljoin(base_url, filename)

    L.info(f"downloading from {url} to {data_path}")
    download_filepath = download_file(url, data_path, filename)

    L.info(f"extracting {download_filepath}")
    extract_with_progress(download_filepath, with_pattern="a3m")
    volume.commit()


def move_to_modal(mmcif_relative_path, s3_dir_path, modal_dir_path):
    def exists_and_same_size(a, b):
        return a.exists() and b.exists() and a.stat().st_size == b.stat().st_size

    retries = 3
    while retries > 0:
        try:
            source_filepath = s3_dir_path / mmcif_relative_path
            dest_filepath   = modal_dir_path / mmcif_relative_path

            if exists_and_same_size(source_filepath, dest_filepath):
                print(f"exists: {mmcif_relative_path.stem} ", end="")
                return True

            dest_filepath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_filepath, dest_filepath)
            print(f"d/l {mmcif_relative_path.stem} ", end="")
            return True
        except Exception as e:
            print("")
            L.info(f"Warning: {e}")
            L.info(f"retrying d/l of {mmcif_relative_path.stem}")
            retries -= 1
    return False

def scan_directory(subdir_path):
    return list([f for f in subdir_path.rglob("*") if f.is_file()])

@app.function(
    image=mmcif_image,
    volumes={
        volume_path: volume,
        s3_bucket_path: modal.CloudBucketMount(
            bucket_name="pdbsnapshots",
            read_only=True
        ),
    },
    memory=8 * GiB,
    timeout=6 * HOURS,
)
def setup_mmcif_database(
    snapshot_id: Path,
    pdb_type: str,
    pdb_port=33444,
    pdb_server="rsync.wwpdb.org::ftp",
    num_workers_per_cpu=8,
):
    import subprocess
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    assert pdb_type in ("divided", "obsolete")

    max_workers = num_workers_per_cpu * os.cpu_count()

    s3_dir_path = (
        s3_bucket_path / snapshot_id / "pub" / "pdb"
        / "data" / "structures" / pdb_type / "mmCIF"
    )
    modal_dir_path = data_path / "pdb" / pdb_type

    L.info(f"scanning: {s3_dir_path} to get total file count")
    mmcif_paths = [f for f in s3_dir_path.iterdir() if f.is_file()]
    s3_subdir_paths = [d for d in s3_dir_path.iterdir() if d.is_dir()]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
         futures = [executor.submit(scan_directory, d) for d in s3_subdir_paths]

         for future in tqdm(futures, desc="scanning s3 directories"):
             mmcif_paths.extend(future.result())

    mmcif_relative_paths = [f.relative_to(s3_dir_path) for f in mmcif_paths]

    tasks_args = [
        (f, s3_dir_path, modal_dir_path) for f in mmcif_relative_paths
    ]

    L.info(f"d/l all PDBs with process pool, max_workers={max_workers},"
          f" {len(tasks_args)} d/l tasks will be started."
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(move_to_modal, *args) for args in tasks_args]
        results = []
        with tqdm(total=len(tasks_args), desc="D/L S3 PDB files") as pbar:
            for future in as_completed(futures):
                results.append(future)
                pbar.update(1)

    L.info(f"finalizying downloads with: {pdb_server}")
    command = [
        "rsync",
        "-rlpt", "-z",
        "--delete",
        f"--port={pdb_port}",
        f"{pdb_server}/data/structures/{pdb_type}/mmCIF",
        f"{modal_dir_path}"
    ]
    subprocess.run(command, check=True)
    volume.commit()

@app.local_entrypoint()
def main(
    uniref_db_name="uniref30_2302",
    metagenomic_db_name="colabfold_envdb_2021",
    pdb_db_name="pdb100_230517",
    pdb_foldseek_db_name="pdb100_foldseek_230517",
    pdb_aws_snapshot: str = "20240101",
    mmseqs_no_index: bool = False, # TODO
    mmseqs_force_merge: bool = True,
):

    colabfold_url = "https://wwwuser.gwdg.de/~compbiol/colabfold/"
    hhsuite_url = (
        "https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/"
    )

    function_calls = [
        setup_profile_database.spawn(
            colabfold_url, uniref_db_name, mmseqs_no_index, mmseqs_force_merge,
        ),
        setup_profile_database.spawn(
            colabfold_url, metagenomic_db_name, mmseqs_no_index, mmseqs_force_merge,
        ),
        setup_fasta_database.spawn(colabfold_url, pdb_db_name, mmseqs_force_merge),
        setup_foldseek_database.spawn(hhsuite_url, pdb_db_name),
        setup_mmcif_database.spawn(pdb_aws_snapshot, "divided"),
        setup_mmcif_database.spawn(pdb_aws_snapshot, "obsolete"),
    ]

    for function_call in function_calls:
        L.info(function_call.get())

# ### Helper Functions

def download_file(url, dest_path, filename):
    import subprocess
    ARIA_NUM_CONNECTIONS = 8
    command = [
       "aria2c",
        "--log-level=warn",
        "-x", str(ARIA_NUM_CONNECTIONS),
        "-o", filename,
        "-c",
        "-d", dest_path,
        url
    ]
    subprocess.run(command, check=True)

    return dest_path / filename


def extract_with_progress(
    filepath,
    with_pattern="",
    chunk_size=1024 * 1024
):
    import tarfile

    from tqdm import tqdm
    assert filepath[-len(".tar.gz"):] == ".tar.gz"

    extraction_filepath = filepath.with_suffix("").with_suffix("")

    extraction_complete_filepath = (
        filepath.with_suffix("").with_suffix(".complete")
    )
    if extraction_complete_filepath.exists():
        L.info("extraction already complete, skipping")
        return extraction_filepath

    mode = "r|*"
    L.info(f"opening with tarfile mode {mode}")
    with tarfile.open(filepath, mode) as tar:
        L.info("opened")
        for member in tar:
            if not member.isfile() or with_pattern not in member.name:
                continue
            member_path = data_path / member.name

            if member_path.exists() and member_path.stat().st_size == member.size:
                L.info(f"already extracted {member.name}, skipping")
                continue

            extract_file = tar.extractfile(member)
            L.info(f"member size: {format_human_readable_bytes(member.size)}")

            file_progress = tqdm(
                total=member.size, unit='B', desc=member.name, unit_scale=True
            )
            with open(member_path, 'wb') as f:
                while True:
                    chunk = extract_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    file_progress.update(len(chunk))

            file_progress.close()

    return extraction_filepath


def format_human_readable_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise Exception("size too large")
