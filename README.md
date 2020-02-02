# Starting kit for the FAN revenue prediction RAMP challenge

[![Build Status](https://travis-ci.org/ramp-kits/fan_revenue_prediction.svg?branch=master)](https://travis-ci.org/ramp-kits/fan_revenue_prediction)

## Getting started

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `jupyter`
* `ramp-workflow`

You can get started on this RAMP challenge with the
[dedicated notebook](FAN_starting_kit.ipynb) by running the following command
from the root directory:

```
jupyter notebook FAN_starting_kit.ipynb
```
## Advanced install using `conda` (optional)

We provide an `environment.yml` file which can be used with `conda` to
create a clean environment and install the necessary dependencies.

```
conda env create -f environment.yml
```

Then, you can activate the environment using:

```
source activate FAN
```

for Linux and MacOS. In Windows, use the following command instead:

```
activate FAN
```

For more information on the [RAMP](http:www.ramp.studio) ecosystem go to
[`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow).