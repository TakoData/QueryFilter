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
pip install git+https://github.com/TakoData/QueryFilter.git@0.1.4
```
## Usage

See the [demo notebook](demo.ipynb) for a more interactive example.

```
from tako_query_filter.filter import TakoQueryFilter

query_filter = TakoQueryFilter()

queries = [
    "aapl vs nvda",
    "trump vs. harris",
    "china gdp",
    "san francisco game this week"
]

query_filter.predict(queries)
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
