# Query Filter
A set of models for filtering out queries that the Tako API won't be able to serve.
Handles pulling 


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
```
pip install git+https://github.com/TakoData/QueryFilter.git
```

## Usage

### Prerequisites
- Get access to Tako HF repos
- Export your `HF_TOKEN` as an environment variable

### Example
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
