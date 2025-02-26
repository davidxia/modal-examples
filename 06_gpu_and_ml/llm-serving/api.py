from pathlib import Path
import modal

import asyncio
import time
from uuid import uuid4
import json

start_time = time.monotonic()  # on remote, time that code started running Modal

here = Path(__file__).parent
deployment_id = uuid4()

trtllm_version = "0.17.0.post1"

if trtllm_version == "0.17.0.post1":
    CUDA_VERSION = "12.8.0"
    GIT_HASH = "258c7540c03517def55d9a5aadfa9288af474e1b"
else:
    raise Exception("Unsupported trtllm version")


N_GPUS = 1
GPU_CONFIG = f"H100:{N_GPUS}"

# MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# FP8 does not work with this model:
# MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct"  # default full precision model

MINUTES = 60  # seconds


app = modal.App("example-trtlllm-v3")

volume = modal.Volume.from_name(
    "example-trtllm-volume-v2", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
DATASETS_PATH = VOLUME_PATH / "dataset"
DATASET_PATH = DATASETS_PATH / "dataset.txt"  # FIXME jsonl

ENGINE_PATHS = [
    # f"/root/{MODEL_ID}/trtllm_engine",
    # f"/root/{MODEL_ID}/trtllm_engine_fp8",
    f"/root/{MODEL_ID}/trtllm_engine_fp8_lookahead",
    ]
# ENGINE_PATH = "/vol/engines/trtllm_engine_fp8_lookahead"

# TODO break into building and saving
def get_configs(engine_path):
    from tensorrt_llm import BuildConfig
    from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig, LookaheadDecodingConfig

    if 'fp8' in engine_path:
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)

        calib_config = CalibConfig(
            calib_batches=512,
            calib_batch_size=1,
            calib_max_seq_length=2048,
            tokenizer_max_seq_length=4096
        )

        build_config = BuildConfig(
            max_input_len=2 ** 15,
            max_num_tokens=2 ** 16,
            max_batch_size=16,
        )
        build_config.plugin_config.multiple_profiles = True
    else:
        quant_config, calib_config, build_config = None, None, None

    if 'lookahead' in engine_path:
        # Examples:
        # https://github.com/NVIDIA/TensorRT-LLM/blob/c366de170f71dd6a350c929ecfdde6a922a05eb1/examples/llm-api/llm_lookahead_decoding.py#L15
        # https://github.com/NVIDIA/TensorRT-LLM/blob/2ea17cdad28bed0f30e80eea5b1380726a7c6493/examples/llm-api/llm_lookahead_decoding.py#L4
        decoding_config = LookaheadDecodingConfig(
            max_window_size=4,
            max_ngram_size=4,
            max_verification_set_size=4,
        )
    else:
        decoding_config = None

    return {
        'quant_config': quant_config,
        'calib_config': calib_config,
        'build_config': build_config,
        'speculative_config': decoding_config
    }

def build_engines():
    from tensorrt_llm import LLM
    from huggingface_hub import snapshot_download

    print(f"downloading base model: {MODEL_ID}")
    snapshot_download(
        MODEL_ID,
        local_dir=MODELS_PATH / MODEL_ID,
    )

    num_engines = len(ENGINE_PATHS)
    for i, engine_path in enumerate(ENGINE_PATHS):
        print(f"{i}/{num_engines}) building new2 engine {engine_path}")
        llm = LLM(
            model=MODELS_PATH / MODEL_ID,
            tensor_parallel_size=N_GPUS,
            **get_configs(engine_path),
        )

        print(f"{i}/{num_engines}) saving engine path to {engine_path}")
        llm.save(engine_path)

        print("deleting llm object to free GPU memory")
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
        build_engines,
        volumes={VOLUME_PATH: volume},
        cpu=16,
        timeout=120*MINUTES,
        gpu=GPU_CONFIG,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

with tensorrt_image.imports():
    from datetime import datetime

    import numpy as np
    import random
    import torch
    import transformers


@app.cls(
    image=tensorrt_image,
    container_idle_timeout=10 * MINUTES,
    volumes={VOLUME_PATH: volume},
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=1,
    concurrency_limit=32,
)
class Model:
    # def __init__(self, engine_path):
        # self.engine_path = engine_path

    @modal.enter()
    def load(self):
        from tensorrt_llm import LLM
        self.engine_path = ENGINE_PATHS[-1]

        configs = get_configs(self.engine_path)

        print(f"loading engine {self.engine_path}")
        self.llm = LLM(
            model=self.engine_path,
            tensor_parallel_size=N_GPUS,
            **configs,
        )

        self.decoding_config = configs['speculative_config']
        self.cold_boot_s = time.monotonic() - start_time

    @modal.method()
    async def generate(self, prompt):
        from tensorrt_llm import SamplingParams
        from dataclasses import asdict
        sampling_params = SamplingParams(
            temperature=0.8, 
            top_p=0.95,
            lookahead_config=self.decoding_config,
        )

        start_time = time.monotonic()

        outputs = self.llm.generate([prompt], sampling_params)
        llm_latency_ms = int(1000 * (time.monotonic() - start_time))

        outputs = [asdict(output.outputs[0]) for output in outputs] # generation samples
        results = {
            "stats": {
                "llm_latency_ms": llm_latency_ms,
                "cold_boot_s": self.cold_boot_s,
            },
            "outputs": outputs,
        }

        return results

    @modal.asgi_app()
    def web(self):
        import io
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse

        webapp = FastAPI()

        @webapp.get("/", response_class=PlainTextResponse)
        @webapp.get("/health", response_class=PlainTextResponse)
        async def health():
            print("health check")
            return "OK"

        @webapp.post("/")
        @webapp.post("/predict")
        async def predict(request: dict):
            from tensorrt_llm import SamplingParams
            from dataclasses import asdict

            prompt = request['prompt']
            sampling_params = SamplingParams(
                temperature=0.8, 
                top_p=0.95,
                lookahead_config=self.decoding_config,
            )

            start_time = time.monotonic()

            outputs = self.llm.generate([prompt], sampling_params)
            llm_latency_ms = int(1000 * (time.monotonic() - start_time))

            outputs = [asdict(output.outputs[0]) for output in outputs] # generation samples
            results = {
                "stats": {
                    "llm_latency_ms": llm_latency_ms,
                    "cold_boot_s": self.cold_boot_s,
                },
                "outputs": outputs,
            }

            return results

        return webapp

    # @moda.web_endpoint(method="POST", docs=True)
    # async def web_generate(self, request: dict):
        # engine_path = ENGINE_PATHS[-1]
        # output = await self.generate.remote.aio(request['prompt'])
        # return output

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
    region='us-east-1',
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
    region='us-east-1',
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
        "--dataset", DATASET_PATH, "--engine_dir", engine_path,
        "--num_requests", "10"])
    print("finished")

@app.local_entrypoint()
async def main(
    prompts_path: str = here / "data" / "raw_samples.jsonl",
    prompt_key: str = "prompt",
    max_prompts: int = None,
    model_name: str = None,
    engine_kwargs_json_path: str = here / "configs" / "inference_config.json",
    experiment_id: str = None,
):
    for engine_path in ENGINE_PATHS[-1:]:
    # for engine_path in ENGINE_PATHS:
        experiment_id = generate_experiment_id()
        print(f"Running experiment {experiment_id}")
        prompts_path = Path(prompts_path)
        engine_kwargs_json_path = Path(engine_kwargs_json_path)

        sample_prompts = load_sample_prompts(prompts_path, prompt_key)[:max_prompts]
        # sample_prompts = expand_prompts_by_n_factor(sample_prompts, 4)
        print (f"Benchmarking {len(sample_prompts)} prompts with {engine_path}")
        engine_kwargs = load_engine_kwargs(engine_kwargs_json_path)

        # model_service = Model(engine_path)
        model_service = Model()
        # model_service = AsyncLLMEngineService(
            # default_sampling_params={"temperature": 0.0, "max_tokens": 1024},
            # model_name=model_name,
            # **engine_kwargs,
        # )
        model_service.generate.remote("gm")  # warmup

        tasks = [fetch(prompt, model_service.generate) for prompt in sample_prompts]

        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            result["engine_path"] = engine_path
            # print(f"Result completed in {result['client_latency_ms']}ms")
            result["experiment_id"] = experiment_id
            results.append(result)

        llm_latencies_ms = [r["stats"]["llm_latency_ms"] for r in results]
        p50 = np.percentile(llm_latencies_ms, 50)
        p99 = np.percentile(llm_latencies_ms, 99)

        save_results(results)
        print(f"Finished experiment {experiment_id} | p50: {p50:.2f}ms / p99: {p99:.2f}ms")


async def fetch(prompt, target):
    start_time = time.monotonic()
    out_handle = target.spawn(prompt)
    result = {
        "prompt_chars": len(prompt),
        "out_handle": out_handle,
        "start_time": start_time,
    }
    response = await result["out_handle"].get.aio()
    end_time = time.monotonic()
    result |= response

    result["client_latency_ms"] = int(1000 * (end_time - start_time))
    result["response_token_count"] = sum(
        len(output["token_ids"]) for output in response["outputs"]
    )
    result["out_handle"] = result["out_handle"].object_id
    result["deployment_id"] = str(deployment_id)

    del result["start_time"]

    return result

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.4", 
        "starlette==0.41.2",
    )
@app.function(
    image=web_image,
    region='us-east-1',
    allow_concurrent_inputs=1000
)
@modal.web_endpoint(method="POST", docs=True)
async def web_generate(request: dict):
    return request['prompt']
    # engine_path = ENGINE_PATHS[-1]
    # output = await Model(engine_path).generate.remote.aio(request['prompt'])
    # return output


def load_sample_prompts(
    path=Path(__file__).parent / "data" / "raw_samples.jsonl", prompt_key="prompt"
):
    prompts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj[prompt_key] if prompt_key else obj)

        return prompts


def load_engine_kwargs(
    path=Path(__file__).parent / "configs" / "inference_config.json",
):
    return json.loads(Path(path).read_text())


def save_results(results, path=Path(__file__).parent / "data" / "results.jsonl"):
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def expand_prompts_by_n_factor(prompts, n):
    "Expand prompts with prefix uniquness to avoid KV cache hits."
    new_prompts = []
    for i in range(n):
        for prompt in prompts:
            new_prompts.append("\n" * i + prompt)
    return new_prompts

def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_experiment_id():
    from wonderwords import RandomWord

    r_gen = RandomWord()
    verb = r_gen.word(
        include_parts_of_speech=["verb"], word_min_length=4, word_max_length=7
    )
    adjective = r_gen.word(
        include_parts_of_speech=["adjective"], word_min_length=4, word_max_length=7
    )
    noun = r_gen.word(
        include_parts_of_speech=["noun"], word_min_length=4, word_max_length=7
    )

    return "-".join([verb, adjective, noun])

def main_og():
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
    # benchmark.remote(ENGINE_PATHS[1], "latency")
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
