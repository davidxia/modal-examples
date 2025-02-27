#"kv_cache_quant_algo": "FP8"
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import os
import time
from uuid import uuid4

import modal
import modal.runner

start_time = time.monotonic()  # on remote, time that code started running Modal

here = Path(__file__).parent
deployment_id = uuid4()

trtllm_version = "0.17.0.post1"
CUDA_VERSION = "12.8.0"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MINUTES = 60  # seconds


app = modal.App("trtllm-benchmark-inference")

volume = modal.Volume.from_name(
    "trtllm-benchmark-inference-volume", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

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
    ).env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(MODELS_PATH), 'TENSORRT_LLM_LOG_LEVEL': 'ERROR'}
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "starlette==0.41.2",
    )
)

with tensorrt_image.imports():
    import numpy as np
    import random
    import torch

# @app.cls(
    # image=tensorrt_image,
    # container_idle_timeout=10 * MINUTES,
    # volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    # gpu="H100:2",
    # cloud="oci",
    # allow_concurrent_inputs=1,
    # concurrency_limit=16
# )
# @app.cls(
    # image=tensorrt_image,
    # container_idle_timeout=10 * MINUTES,
    # volumes={VOLUME_PATH: volume},
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    # **infrastructure_kwargs
# )
class AsyncTRTLLMEngineService:
    # engine_kwargs_string: str = modal.parameter()
    def __init__(self, engine_kwargs_string):
        self.engine_kwargs_string = engine_kwargs_string

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download

        from tensorrt_llm import BuildConfig
        from tensorrt_llm.llmapi import (
            QuantConfig, QuantAlgo, KvCacheConfig, CalibConfig, LookaheadDecodingConfig
        )
        from tensorrt_llm.plugin.plugin import PluginConfig
        from tensorrt_llm import LLM

        # self.engine_kwargs_string = '{"tensor_parallel_size": 2}'
        engine_kwargs = json.loads(self.engine_kwargs_string)
        print("Number of GPUs:", engine_kwargs["tensor_parallel_size"])

        engine_folder_name = hash_config_string(json.dumps(engine_kwargs, sort_keys=True))
        print(f"engine config name: {engine_folder_name}")

        seed_everything()

        print("downloading base model if necessary")
        snapshot_download(
            MODEL_ID,
            local_dir=MODELS_PATH / MODEL_ID,
        )

        print("building TRTLLM engine config objects")
        for quant_key in engine_kwargs["quant_config"].keys():
            print(f"{quant_key}: {engine_kwargs['quant_config'][quant_key]}")
            engine_kwargs["quant_config"][quant_key] = (
                QuantAlgo[engine_kwargs["quant_config"][quant_key]]
            )
        engine_kwargs["quant_config"] = QuantConfig(**engine_kwargs["quant_config"])

        engine_kwargs["kv_cache_config"] = KvCacheConfig(**engine_kwargs["kv_cache_config"])

        engine_kwargs["calib_config"] = CalibConfig(**engine_kwargs["calib_config"])

        engine_kwargs["build_config"]["plugin_config"] = (
            PluginConfig.from_dict(engine_kwargs["build_config"]["plugin_config"])
        )
        engine_kwargs["build_config"] = BuildConfig(**engine_kwargs["build_config"])

        # TODO: Add support for other decoding configs
        engine_kwargs["speculative_config"] = (
            LookaheadDecodingConfig(**engine_kwargs["speculative_config"])
        )

        engine_path = MODELS_PATH / MODEL_ID / "trtllm_engine" / engine_folder_name
        if not os.path.exists(engine_path):
            print(f"building new engine at {engine_path}")
            llm = LLM(model=MODELS_PATH / MODEL_ID, **engine_kwargs)
            llm.save(engine_path)
            del llm

        print (f"loading engine from {engine_path}")
        self.llm = LLM(model=engine_path, **engine_kwargs)

        self.lookahead_config = engine_kwargs["speculative_config"]

        self.cold_boot_s = time.monotonic() - start_time

    @modal.method()
    async def generate(self, prompt):
        from tensorrt_llm import SamplingParams
        from dataclasses import asdict
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            lookahead_config=self.lookahead_config,
        )

        start_time = time.monotonic()
        output = await self.llm.generate_async(prompt, sampling_params)
        llm_latency_ms = int(1000 * (time.monotonic() - start_time))

        outputs = [asdict(output.outputs[0])] # generation samples
        results = {
            "stats": {
                "llm_latency_ms": llm_latency_ms,
                "cold_boot_s": self.cold_boot_s,
            },
            "outputs": outputs,
        }

        return results

    @modal.method()
    async def noop(self):
        return {time.monotonic()}

    @modal.web_endpoint(method="POST", docs=True)
    async def web_generate(self, request: dict):
        return self.generate.local(request['prompt'])

    @modal.exit()
    def shutdown(self):
        del self.llm

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str, default=here / "data" / "raw_samples.jsonl")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--engine_kwargs_json_path", type=str, default=here / "configs" / "trtllm_inference_config.json")
    parser.add_argument("--infrastructure_kwargs_json_path", type=str, default=here / "configs" / "infrastructure_config.json")
    parser.add_argument("--save_results_path", type=str, default=here / "data" / "results.jsonl")
    args = parser.parse_args()

    prompts_path = args.prompts_path
    prompt_key = args.prompt_key
    max_prompts = args.max_prompts
    engine_kwargs_json_path = args.engine_kwargs_json_path
    infrastructure_kwargs_json_path = args.infrastructure_kwargs_json_path
    save_results_path = args.save_results_path

    infrastructure_kwargs = json.loads(Path(infrastructure_kwargs_json_path).read_text())

    prompts_path = args.prompts_path
    prompt_key = args.prompt_key
    max_prompts = args.max_prompts
    engine_kwargs_json_path = args.engine_kwargs_json_path
    save_results_path = args.save_results_path

    model_class = app.cls(
        image=tensorrt_image,
        container_idle_timeout=10 * MINUTES,
        volumes={VOLUME_PATH: volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
        **infrastructure_kwargs
    )(AsyncTRTLLMEngineService)

    with modal.enable_output():
        with app.run():
            experiment_id = generate_experiment_id()
            print(f"Running experiment {experiment_id}")
            prompts_path = Path(prompts_path)
            engine_kwargs_json_path = Path(engine_kwargs_json_path)

            sample_prompts = load_sample_prompts(prompts_path, prompt_key)[:max_prompts]
            print(f"Benchmarking {len(sample_prompts)} prompts")
            print(f"\tEngine Config: {engine_kwargs_json_path}")
            print(f"\tInfrastructure Config: {infrastructure_kwargs_json_path}")
            engine_kwargs = json.loads(Path(engine_kwargs_json_path).read_text())
            engine_kwargs["tensor_parallel_size"] = get_gpu_count(infrastructure_kwargs["gpu"])
            engine_kwargs_string = json.dumps(engine_kwargs)


            # AsyncTRTLLMEngineService       
            model_service = model_class(engine_kwargs_string=engine_kwargs_string)
            await model_service.generate.remote.aio("gm")  # warmup

            tasks = [fetch(prompt, model_service.generate) for prompt in sample_prompts]

            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                result["inference_config"] = engine_kwargs
                result["infrastructure_config"] = infrastructure_kwargs
                # print(f"Result completed in {result['client_latency_ms']}ms")
                result["experiment_id"] = experiment_id
                results.append(result)

            save_results(save_results_path, results)

            print(f"Finished experiment {experiment_id} | sample_size={len(sample_prompts)}")
            for (key_name, get_value) in (
                ("llm_latency_ms", lambda x: x["stats"]["llm_latency_ms"]),
                ("client_latency_ms", lambda x: x["client_latency_ms"]),
            ):
                stats = [get_value(r) for r in results]
                p50 = np.percentile(stats, 50)
                p99 = np.percentile(stats, 99)

                print(f"\t {key_name:<20}: p50: {p50:>6.2f}ms / p99: {p99:>6.2f}ms")


def get_gpu_count(gpu_string):
    if ":" not in gpu_string:
        return 1
    return int(gpu_string[gpu_string.index(":") + 1:])


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

def load_sample_prompts(path, prompt_key="prompt"):
    prompts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj[prompt_key] if prompt_key else obj)

        return prompts

def save_results(path, results):
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_experiment_id():
    from wonderwords import RandomWord

    r_gen = RandomWord()
    return "-".join(
        [r_gen.word(include_parts_of_speech=[x], word_min_length=4, word_max_length=7)
        for x in ["verb", "adjective", "noun"]]
    )

def hash_config_string(config_string):
    return hashlib.md5(config_string.encode()).hexdigest()

if __name__ == "__main__":
    asyncio.run(main())
