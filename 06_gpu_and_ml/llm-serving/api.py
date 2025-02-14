from pathlib import Path
import modal

trtllm_version = "0.17.0.post1"

if trtllm_version == "0.17.0.post1":
    CUDA_VERSION = "12.8.0"
    GIT_HASH = "258c7540c03517def55d9a5aadfa9288af474e1b"
else:
    raise Exception("Unsupported trtllm version")


N_GPUS = 1
GPU_CONFIG = f"H100:{N_GPUS}"

# MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct"  # default full precision model

MINUTES = 60  # seconds


app = modal.App("example-trtlllm-v2")

volume = modal.Volume.from_name(
    "example-trtllm-volume-v2", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
DATASETS_PATH = VOLUME_PATH / "dataset"
DATASET_PATH = DATASETS_PATH / "dataset.txt"  # FIXME jsonl

ENGINE_PATHS = [
    f"/root/{MODEL_ID}/trtllm_engine",
    # f"/root/{MODEL_ID}/trtllm_engine_fp8",
    # f"/root/{MODEL_ID}/trtllm_engine_fp8_lookahead",
    ]
# ENGINE_PATH = "/vol/engines/trtllm_engine_fp8_lookahead"

# TODO break into building and saving
def build_engine():
    from tensorrt_llm import LLM, BuildConfig
    from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig, LookaheadDecodingConfig

    num_engines = len(ENGINE_PATHS)
    for i, engine_path in enumerate(ENGINE_PATHS):
        if 'fp8' in engine_path:
            quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)

            calib_config = CalibConfig(
                calib_batches=512,
                calib_batch_size=1,
                calib_max_seq_length=2048,
                tokenizer_max_seq_length=4096
            )

            build_config = BuildConfig(
                max_num_tokens=2048,
                max_batch_size=512,
            )
        else:
            quant_config, calib_config, build_config = None, None, None

        decoding_config = None
        if 'lookahead' in engine_path:
            # https://github.com/NVIDIA/TensorRT-LLM/blob/c366de170f71dd6a350c929ecfdde6a922a05eb1/examples/llm-api/llm_lookahead_decoding.py#L15
            decoding_config = LookaheadDecodingConfig(
                max_window_size=4,
                max_ngram_size=4,
                max_verification_set_size=4,
            )

        print(f"{i}/{num_engines}) building engine {engine_path}")
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=N_GPUS,
            quant_config=quant_config,
            calib_config=calib_config,
            build_config=build_config,
            speculative_config=decoding_config,
        )

        print(f"{i}/{num_engines}) saving engine path to {engine_path}")
        llm.save(engine_path)
        print(f"{i}/{num_engines}) saved engine path to {engine_path}, deleting llm")
        del llm


tensorrt_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.10
    ).entrypoint(
        [] # remove verbose logging by base image on entry
    ).apt_install(
        "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
    ).pip_install(
        f"tensorrt-llm=={trtllm_version}",
        "pynvml<12",  # avoid breaking change to pynvml version API
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    ).pip_install(
        "hf-transfer==0.1.9",
        "huggingface_hub==0.28.1",
    ).env(  # hf-transfer for faster downloads
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(MODELS_PATH),
        }
    ).run_function(
        build_engine,
        volumes={VOLUME_PATH: volume},
        cpu=16,
        timeout=120*MINUTES,
        gpu=GPU_CONFIG,
    )
)

@app.cls(
    image=tensorrt_image,
    container_idle_timeout=10 * MINUTES,
    volumes={VOLUME_PATH: volume},
    gpu=GPU_CONFIG,
)
class Model:
    @modal.enter()
    def load(self, engine_path):
        from tensorrt_llm import LLM
        print(f"loading engine {engine_path}")
        self.llm = LLM(model=engine_path)

    @modal.method()
    def inference(self, prompts):
        from tensorrt_llm import SamplingParams
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = self.llm.generate(prompts, sampling_params)

        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

    @modal.exit()
    def shutdown(self):
        del self.llm

def get_model_path():
    # Since we don't know the snapshot_id we glob to find it.
    from pathlib import Path
    import glob
    return glob.glob(
        str(
            MODELS_PATH / ("hub/models--" + MODEL_ID.replace("/", "--"))
            / "snapshots" / "*"
        )
    )[0]

@app.function(
    image=tensorrt_image,
    container_idle_timeout=10 * MINUTES,
    keep_warm=1,
    volumes={VOLUME_PATH: volume},
    gpu=GPU_CONFIG,
)
def prepare_dataset(
    kwargs: str = "--stdout --tokenizer _MODEL_PATH_" 
    " token-norm-dist --input-mean 2048 --output-mean 2048"
    " --input-stdev 0 --output-stdev 0 --num-requests 1000"
):
    import os
    import subprocess

    kwargs = kwargs.replace("_MODEL_PATH_", get_model_path())

    subprocess.run(["git", "clone", "https://github.com/NVIDIA/TensorRT-LLM"])
    subprocess.run(["git", "checkout", GIT_HASH], cwd="TensorRT-LLM")

    os.makedirs(DATASETS_PATH, exist_ok=True)
    with open(DATASET_PATH, 'w') as f:
        subprocess.run(
            ["python", "prepare_dataset.py"] + kwargs.split(" "),
            cwd="TensorRT-LLM/benchmarks/cpp",
            stdout=f,
            text=True,
        )

    # breakpoint()

@app.function(
    image=tensorrt_image,
    container_idle_timeout=10 * MINUTES,
    timeout=120 * MINUTES,
    volumes={VOLUME_PATH: volume},
    gpu=GPU_CONFIG,
)
# cmd = "trtllm-bench --model /vol/models/hub/models--unsloth--Qwen2.5-Coder-7B-Instruct/snapshots/3fd3aab092612530a892ff49027dfd4f39046ec3 latency --dataset /vol/dataset/dataset.txt --engine_dir /root/unsloth/Qwen2.5-Coder-7B-Instruct/trtllm_engine"
def benchmark(engine_path, mode: str = "latency"):
    import subprocess

    assert mode in ["latency", "throughput"]

    # cmd = f"trtllm-bench --model {get_model_path()} {mode} --dataset {DATASET_PATH} --engine_dir {engine_path}"
    # print(f"running {cmd}")
    # subprocess.run(cmd, shell=True)
    # subprocess.run(cmd.split(" "))

    # print ('return')
    # return

    print(f"running trtllm-bench with {engine_path} and mode={mode}")
    subprocess.run(["trtllm-bench", "--model", get_model_path(), mode,
        "--dataset", DATASET_PATH, "--engine_dir", engine_path])
    print("finished")


@app.local_entrypoint()
def main():
    code_prompts = [
        "for i",
        "int x = 1\n",
        "import num",
        "while",
    ]
    language_prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # prepare_dataset.remote()
    # print (ENGINE_PATHS)
    print('remote')
    # benchmark.remote()
    # benchmark.remote(ENGINE_PATHS[0])
    benchmark.remote(ENGINE_PATHS[0], "latency")
    print ('done')
    return

"""
def pass():
    return
    for result in benchmark.starmap([(e, "latency") for e in ENGINE_PATHS]):
        print(result)
    return
    # fc = benchmark.spawn("throughput")
    # print(fc)
    # fc.get()

    # generated_texts = Model().inference.remote(language_prompts)
    generated_texts = Model(ENGINE_PATHS[0]).inference.remote(code_prompts)
    for i, text in enumerate(generated_texts):
        print(f"{i}) {text}")
"""
