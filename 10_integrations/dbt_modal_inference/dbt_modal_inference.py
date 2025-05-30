# # LLM inference within your data warehouse using dbt python models

# In this example we demonstrate how you could combine [dbt's python models](https://docs.getdbt.com/docs/build/python-models)
# with LLM inference models powered by Modal, allowing you to run serverless gpu workloads within dbt.

# This example runs [dbt](https://docs.getdbt.com/docs/introduction) with a [DuckDB](https://duckdb.org)
# backend directly on top of Modal, but could be translated to run on any dbt-compatible
# database that supports python models. Similarly you could make these requests from UDFs
# directly in SQL instead if you don't want to use dbt's python models.

# In this example we use an LLM deployed in a previous example: [Serverless TensorRT-LLM (LLaMA 3 8B)](https://modal.com/docs/examples/trtllm_llama)
# but you could easily swap this for whichever Modal Function you wish. We use this to classify the sentiment
# for free-text product reviews and aggregate them in subsequent dbt sql models. These product names, descriptions and reviews
# were also generated by an LLM running on Modal!

# ## Configure Modal and dbt

# We set up the environment variables necessary for dbt and
# create a slim debian and install the packages necessary to run.

import pathlib

import modal

LOCAL_DBT_PROJECT = (  # local path
    pathlib.Path(__file__).parent / "dbt_modal_inference_proj"
)
PROJ_PATH = "/root/dbt"  # remote paths
VOL_PATH = "/root/vol"
DB_PATH = f"{VOL_PATH}/db"
PROFILES_PATH = "/root/dbt_profile"
TARGET_PATH = f"{VOL_PATH}/target"

# We also define the environment our application will run in --
# a container image, similar to Docker.
# See [this guide](https://modal.com/docs/guide/custom-container) for details.

dbt_image = (  # start from a slim Linux image
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(  # install python packages
        "dbt-duckdb==1.8.1",  # dbt with duckdb connector
        "pandas==2.2.2",  # dataframes
        "pyarrow==17.0.0",  # columnar data lib
        "requests==2.32.3",  # http library
    )
    .env(  # configure dbt environment variables
        {
            "DBT_PROJECT_DIR": PROJ_PATH,
            "DBT_PROFILES_DIR": PROFILES_PATH,
            "DBT_TARGET_PATH": TARGET_PATH,
            "DB_PATH": DB_PATH,
        }
    )
    # We add the local code and configuration into the image
    # so that it will be available when we run dbt
    .add_local_dir(LOCAL_DBT_PROJECT, remote_path=PROJ_PATH)
    .add_local_file(
        local_path=LOCAL_DBT_PROJECT / "profiles.yml",
        remote_path=f"{PROFILES_PATH}/profiles.yml",
    )
)

app = modal.App("duckdb-dbt-inference", image=dbt_image)


# Create a modal.Volume so that we can persist our data
dbt_vol = modal.Volume.from_name("dbt-inference-vol", create_if_missing=True)

# ## Run dbt in a serverless Modal Function

# With Modal it's easy to run python code serverless
# and with dbt's [programmatic invocations](https://docs.getdbt.com/reference/programmatic-invocations)
# you can easily run dbt from python instead of using the command line

# Using the above configuration we can invoke dbt from Modal
# and use this to run transformations in our warehouse.

# The `dbt_run` function does a few things, it:

# 1. creates the directories for storing the DuckDB database and dbt target files

# 2. gets a reference to a deployed Modal Function that serves an LLM inference endpoint

# 3. runs dbt with a variable for the inference url

# 4. prints the output of the final dbt table in the DuckDB parquet output


@app.function(
    volumes={VOL_PATH: dbt_vol},
)
def dbt_run() -> None:
    import os

    import duckdb
    from dbt.cli.main import dbtRunner

    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(TARGET_PATH, exist_ok=True)

    # Remember to either deploy the llama dependency app in your environment
    # first, or change this to use another web endpoint you have:
    ref = modal.Function.from_name(
        "example-trtllm-Meta-Llama-3-8B-Instruct", "generate_web"
    )

    res = dbtRunner().invoke(
        ["run", "--vars", f"{{'inference_url': '{ref.get_web_url()}'}}"]
    )
    if res.exception:
        print(res.exception)

    duckdb.sql(
        f"select * from '{DB_PATH}/product_reviews_sentiment_agg.parquet';"
    ).show()


# Running the Modal Function with

# ```sh
# modal run dbt_modal_inference.py
# ```

# will result in something like:

# ```
# 21:25:21  Running with dbt=1.8.4
# 21:25:21  Registered adapter: duckdb=1.8.1
# 21:25:23  Found 5 models, 2 seeds, 6 data tests, 2 sources, 408 macros
# 21:25:23
# 21:25:23  Concurrency: 1 threads (target='dev')
# 21:25:23
# 21:25:23  1 of 5 START sql table model main.stg_products ................................. [RUN]
# 21:25:23  1 of 5 OK created sql table model main.stg_products ............................ [OK in 0.22s]
# 21:25:23  2 of 5 START sql table model main.stg_reviews .................................. [RUN]
# 21:25:23  2 of 5 OK created sql table model main.stg_reviews ............................. [OK in 0.17s]
# 21:25:23  3 of 5 START sql table model main.product_reviews .............................. [RUN]
# 21:25:23  3 of 5 OK created sql table model main.product_reviews ......................... [OK in 0.17s]
# 21:25:23  4 of 5 START python external model main.product_reviews_sentiment .............. [RUN]
# 21:25:32  4 of 5 OK created python external model main.product_reviews_sentiment ......... [OK in 8.83s]
# 21:25:32  5 of 5 START sql external model main.product_reviews_sentiment_agg ............. [RUN]
# 21:25:32  5 of 5 OK created sql external model main.product_reviews_sentiment_agg ........ [OK in 0.16s]
# 21:25:32
# 21:25:32  Finished running 3 table models, 2 external models in 0 hours 0 minutes and 9.76 seconds (9.76s).
# 21:25:33
# 21:25:33  Completed successfully
# 21:25:33
# 21:25:33  Done. PASS=5 WARN=0 ERROR=0 SKIP=0 TOTAL=5
# ┌──────────────┬──────────────────┬─────────────────┬──────────────────┐
# │ product_name │ positive_reviews │ neutral_reviews │ negative_reviews │
# │   varchar    │      int64       │      int64      │      int64       │
# ├──────────────┼──────────────────┼─────────────────┼──────────────────┤
# │ Splishy      │                3 │               0 │                1 │
# │ Blerp        │                3 │               1 │                1 │
# │ Zinga        │                2 │               0 │                0 │
# │ Jinkle       │                2 │               1 │                1 │
# │ Flish        │                2 │               2 │                1 │
# │ Kablooie     │                2 │               1 │                1 │
# │ Wizzle       │                2 │               1 │                0 │
# │ Snurfle      │                2 │               1 │                0 │
# │ Glint        │                2 │               0 │                0 │
# │ Flumplenook  │                2 │               1 │                1 │
# │ Whirlybird   │                2 │               0 │                1 │
# ├──────────────┴──────────────────┴─────────────────┴──────────────────┤
# │ 11 rows                                                    4 columns │
# └──────────────────────────────────────────────────────────────────────┘
# ```

# Here we can see that the LLM classified the results into three different categories
# that we could then aggregate in a subsequent sql model!

# ## Python dbt model

# The python dbt model in [`dbt_modal_inference_proj/models/product_reviews_sentiment.py`](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/dbt_modal_inference/dbt_modal_inference_proj/models/product_reviews_sentiment.py) is quite simple.

# It defines a python dbt model that reads a record batch of product reviews,
# generates a prompt for each review and makes an inference call to a Modal Function
# that serves an LLM inference endpoint. It then stores the output in a new column
# and writes the data to a parquet file.

# And it's that simple to call a Modal web endpoint from dbt!

# ## View the stored output

# Since we're using a [Volume](https://modal.com/docs/guide/volumes) for storing our dbt target results
# and our DuckDB parquet files
# you can view the results and use them outside the Modal Function too.

# View the target directory by:
# ```sh
# modal volume ls dbt-inference-vol target/
#            Directory listing of 'target/' in 'dbt-inference-vol'
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
# ┃ Filename                      ┃ Type ┃ Created/Modified      ┃ Size      ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
# │ target/run                    │ dir  │ 2024-07-19 22:59 CEST │ 14 B      │
# │ target/compiled               │ dir  │ 2024-07-19 22:59 CEST │ 14 B      │
# │ target/semantic_manifest.json │ file │ 2024-07-19 23:25 CEST │ 234 B     │
# │ target/run_results.json       │ file │ 2024-07-19 23:25 CEST │ 10.1 KiB  │
# │ target/manifest.json          │ file │ 2024-07-19 23:25 CEST │ 419.7 KiB │
# │ target/partial_parse.msgpack  │ file │ 2024-07-19 23:25 CEST │ 412.7 KiB │
# │ target/graph_summary.json     │ file │ 2024-07-19 23:25 CEST │ 1.4 KiB   │
# │ target/graph.gpickle          │ file │ 2024-07-19 23:25 CEST │ 15.7 KiB  │
# └───────────────────────────────┴──────┴───────────────────────┴───────────┘
# ```

# And the db directory:
# ```sh
# modal volume ls dbt-inference-vol db/
#                   Directory listing of 'db/' in 'dbt-inference-vol'
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
# ┃ Filename                                 ┃ Type ┃ Created/Modified      ┃ Size    ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
# │ db/review_sentiments.parquet             │ file │ 2024-07-19 23:25 CEST │ 9.6 KiB │
# │ db/product_reviews_sentiment_agg.parquet │ file │ 2024-07-19 23:25 CEST │ 756 B   │
# └──────────────────────────────────────────┴──────┴───────────────────────┴─────────┘
# ```
#
