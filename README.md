# Query Filter
This library wraps a set of models that predict whether Tako will be able to serve a given query.
It handles downloading the models from Hugging Face, as well as providing a simple API for filtering 
queries.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
```
pip install git+https://github.com/TakoData/QueryFilter.git
```
With embeddings support, which pulls in the large `sentence-transformers` package:
```
pip install "git+https://github.com/TakoData/QueryFilter.git#egg=tako-query-filter[embeddings]"
```

## Usage

### Prerequisites
- Get access to Tako Hugging Face repositories
- Install and init `git-lfs`
- Log into Hugging Face using `huggingface-cli login`

### Examples
See the [demo notebook](demo.ipynb) for a more interactive example.

```
from tako_query_filter.filter import TakoQueryFilter

query_filter = TakoQueryFilter.load_from_hf()

queries = [
    "aapl vs nvda",
    "trump vs. harris",
    "china gdp",
    "san francisco game this week"
]

query_filter.predict(queries)
```
You can also use the `force_download` flag to download the latest version of the models from HuggingFace.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
