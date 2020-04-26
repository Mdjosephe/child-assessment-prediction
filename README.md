

# Members of the team are 2020 data science master at polytechnique (france) studens :
* Loïc Omnès
* Meriem Elmrini
* Thi Hai Yen VU
* Adam Haytoumi
* Mohamed Abdel Wedoud


# Warning : 
the evaluation of models is done with Kappa Metric please see the Kaggle challenge page. Thus if you have a 0.65 score it doesn't necessarely mean that you got 65% of the predictions right !!  You may have guessed even 95% right and get only a 0.56 score with this metric.

# Starting kit for the child assessment prediction RAMP challenge

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
[dedicated notebook](Childstein_starting_kit.ipynb) by running the following command
from the root directory:

```
jupyter notebook Childstein_starting_kit.ipynb
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


