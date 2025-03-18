import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory  # noqa

import modal

logger = logging.getLogger(name=__name__)

MODAL_APP_NAME = "mmseqs-server-app"

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .run_commands(
        [
            "apt-get -qq update && apt-get -qq install -y wget tar",
            # Get MMseqs2 GPU version, release 17
            "wget https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-linux-gpu.tar.gz",
            "mkdir -p /usr/local/mmseqs/",
            "tar -C /usr/local/mmseqs -xzf mmseqs-linux-gpu.tar.gz --strip-components 1",
            "rm mmseqs-linux-gpu.tar.gz",
        ]
    )
    .pip_install("pandas")
)
with image.imports():
    import pandas as pd


vol = modal.Volume.from_name("mmseqs-colabfold-data-advay", create_if_missing=True)
shariq_vol = modal.Volume.from_name("example-compbio-colab-v3", create_if_missing=True)

app = modal.App(MODAL_APP_NAME, image=image)


@app.cls(
    image=image,
    cpu=4,
    memory=(16 * 1024, 64 * 1024),
    gpu="L40S",  # env db requires 40GB GPU; uniref30 requires 10GB
    volumes={"/data": vol, "/shariq": shariq_vol},
    timeout=8 * 60 * 60,
    ephemeral_disk=768 * 1024,
    scaledown_window=60 * 2,  # Keep alive for 2 min
    max_containers=10,
)
class MmseqsServer:
    """
    Some performance characteristics:
    - Queries take 2-3 min on cold start, but are much faster (~30s) subsequently
    """

    mmseqs_binary = Path("/usr/local/mmseqs/bin/mmseqs")

    @modal.enter()
    def startup(self):
        """Startup; index all databases."""
        assert self.mmseqs_binary.exists()
        self.create_databases()
        for db in self.databases():
            self.index_database(db)

    def create_databases(self):
        envdb = "envdb/colabfold_envdb_202108_gpu_db"
        uniref30 = "uniref/uniref30_2302_gpu_db"

        database_suffixes = [envdb, uniref30]
        databases = [f"/data/{db}" for db in database_suffixes]
        for d in range(len(databases)):
            database_suffix = database_suffixes[d]
            tsv_name = database_suffix.split("/")[-1].replace("_gpu_db", "")
            database = databases[d]
            if not Path(database).exists():
                import subprocess
                cmd = f"{self.mmseqs_binary} tsv2exprofiledb /shariq/data/{tsv_name} {database} --gpu 1"
                print (cmd)
                subprocess.run(cmd, shell=True, check=True)


    @staticmethod
    def databases() -> list[str]:
        """Databases available in the server."""
        # All databases should be GPU-friendly padded versions
        # Creation command (run separately then uploaded):
        # mmseqs tsv2exprofiledb "colabfold_envdb_202108" "colabfold_envdb_202108_db" --gpu 1
        envdb = "envdb/colabfold_envdb_202108_gpu_db"
        # Creation command (run separately then uploaded):
        # mmseqs tsv2exprofiledb "uniref30_2302" "uniref30_2302_db" --gpu 1
        uniref30 = "uniref/uniref30_2302_gpu_db"

        # Gather, check existence in data volume, return
        database_suffixes = [envdb, uniref30]
        databases = [f"/data/{db}" for db in database_suffixes]
        assert all(Path(db).exists() for db in databases)
        return databases

    @staticmethod
    def index_database(db: str):
        """Index the given database; short circuits if index already exists."""
        db_path = Path(db)
        assert db_path.exists()

        cmd = [
            MmseqsServer.mmseqs_binary.absolute().as_posix(),
            "createindex",
            db,
            "/tmp/mmseqs",
            "--index-subset",
            "2",
            "--remove-tmp-files",
            "1",
            "--threads",
            "32",
            "--check-compatible",  # Check if recreating is needed
            "1",
        ]
        logger.info(f"Indexing {db}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        vol.commit()  # Commit the index files to the volume

    @modal.method()
    def run_query(self, query: str) -> pd.DataFrame:
        hit_lines: list[str] = []
        for database in self.databases():
            contents = _run_easy_search(
                sequence=query,
                db=database,
                mmseqs_bin=self.mmseqs_binary.absolute().as_posix(),
                use_gpu=True,
                num_threads=32,
            )
            hit_lines.extend(contents)

        return pd.DataFrame(
            [line.split("\t") for line in hit_lines if line], columns=EASY_SEARCH_COLS
        )


EASY_SEARCH_COLS = [
    "query",
    "target",
    "evalue",
    "qstart",
    "qend",
    "qlen",
    "qaln",
    "taln",
]


def _run_easy_search(
    sequence: str, db: str, mmseqs_bin: str, use_gpu: bool = True, num_threads: int = 16
) -> list[str]:
    """Run easy-search on the given sequence, returning contents of resulting file.

    Runtime for a single sequence query on colabfold env db:
    - A100: 30 seconds
    - 16 CPU cores: 5 min
    - 60 CPU cores: 2 min

    Memory consumption (RAM if CPU, GPU memory if GPU):
    - Colabfold env db: ~40GB
    - UniRef30 DB: ~10GB
    """
    # Hacky solution to get clean delims in the outputs; otherwise there are issues with
    # newlines appearing unexpectedly in the middle of lines.
    manual_delim = "$&^$"

    with TemporaryDirectory() as tmpdir:
        query_fasta = Path(tmpdir) / "query.fa"
        query_fasta.write_text(f">{manual_delim}query\n{sequence}")

        outfile = Path(tmpdir) / "search_output"

        cmd = [
            mmseqs_bin,
            "easy-search",
            query_fasta.as_posix(),
            db,
            outfile.as_posix(),
            (Path(tmpdir) / "work").as_posix(),
            "--db-load-mode",
            "2",
            "--num-iterations",
            "3",
            "-a",
            "-e",
            "0.1",
            "--max-seqs",
            "10000",
            "--prefilter-mode",
            "1",
            "--threads",
            str(num_threads),
            "--format-output",
            ",".join(EASY_SEARCH_COLS),
        ]
        if use_gpu:
            cmd.extend(["--gpu", "1"])
        subprocess.run(cmd, check=True)

        assert outfile.exists() and outfile.stat().st_size > 0
        contents = (
            outfile.read_text().replace("\x00", "").replace("\n", "").replace("\r", "")
        ).split(manual_delim)

    return contents


@app.function(timeout=8 * 60 * 60)
def server_easy_search(protein_sequence: str) -> pd.DataFrame:
    """Run easy-search on the given protein sequence."""
    server = MmseqsServer()
    requests = server.run_query.spawn(protein_sequence)
    responses = requests.get()
    return responses

@app.local_entrypoint()
def main():
    server_easy_search.remote("MSSYVYDTRKQVYKQYKQVYKQYKQVYKQYKQVYKQ")
