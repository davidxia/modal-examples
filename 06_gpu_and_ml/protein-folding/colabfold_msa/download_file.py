import logging
import subprocess
from pathlib import Path
import modal

logger = logging.getLogger(name=__name__)

# Create Modal app and volume
app = modal.App("download-file-app")
vol = modal.Volume.from_name("mmseqs-colabfold-data-advay")

# Create base image with required tools
image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .run_commands([
        "apt-get update && apt-get install -y wget unzip"
    ])
)

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600
)
def download_and_unzip(url: str, target_path: str):
    """
    Downloads a file from url and unzips it into the volume at target_path
    
    Args:
        url: URL to download from
        target_path: Path within the volume to save to
    """
    target_path = Path("/data") / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # # Download file
    # logger.info(f"Downloading {url} to {target_path}")
    # subprocess.run(["wget", "-O", f"{target_path}.zip", url], check=True)
    
    # Unzip file
    logger.info(f"Unzipping to {target_path}")
    subprocess.run(["tar", "-xzf", f"{target_path}.zip", "-C", str(target_path.parent)], check=True)
    
    # Remove zip file
    Path(f"{target_path}.zip").unlink()
    
    # Commit changes to volume
    vol.commit()
    
    return str(target_path)

@app.local_entrypoint()
def main(url: str, target_path: str):
    """
    Example usage:
    modal run download_file.py --url https://example.com/file.zip --target_path my/path
    """
    result = download_and_unzip.remote(url, target_path)
    print(f"Downloaded and unzipped to: {result}")
