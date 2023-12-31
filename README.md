# trendalation

An anomaly detection package for all sorts of trends & time series data. The library leverages algorithms like procrustes analysis to compare and contrast the general shape & trajectory of different trends. The **procrustses analysis** technique could be leveraged to get the minimum possible mean squared error between 2 distributions (after mathematical transformations). Using the training dataset, an ideal reference trace is determined for comparision along with a classification threshold from the generated error distribution using procrustes against this reference curve.

<img width="718" alt="before_proc_demo" src="https://github.com/kartikeysinha/trendalation/assets/44055129/54e6dc93-33b8-4890-a086-5e624aff1dc3">
<img width="718" alt="after_proc_demo" src="https://github.com/kartikeysinha/trendalation/assets/44055129/5802bb69-9695-4403-9d9e-1f8c02f59ce3">

## Why Trendalation?

In the realm of data analysis and business intelligence, identifying anomalies within time-series data is paramount. Whether it's monitoring website traffic, financial metrics, or industrial sensor readings, uncovering unusual patterns can be the key to proactive decision-making. Trendalation empowers you to effortlessly build robust anomaly detection models that adapt to your data's nuances.

### Key Features

- **Highly Customizable:** Fine-tune models using several parameters to optimize for your project constraints.
- **Lightweight Solution:** Unlike other solutions, Trendalation provides an extremely lightweight & fast solution, best suited for projects where performance is paramount.
- **Versatile Algorithms:** Spatial analysis makes the algorithms extremely effective for all sorts of trends.

## Installation

### Dependencies

trendalation requires:

- Python (>= 3.8)
- NumPy (>= 1.25.0)
- Scikit-Learn (>=1.3.0)
- SciPy (>= 1.11.1)

### Using pip

The easiest way to install trendalation is using pip:
`pip install trendalation`

## Documentation

### Usage

To train a classifier on a collection of traces, a classifier can be trained on the trace distribution. The model sets up the necessary classification criterion by analyzing various population statistics and trends. eg-

```python
from trendalation.classification import ProcClassifier
clf = ProcClassifier()
clf.fit()
```

To futher optimize the process to the classification criterion determination, additional (optional) training parameters can be provided-

- **y:** The target values (binary class labels) as integers with 1 representing an anomaly. If provided, classifier is fitted only on "good" samples in an attempt to get higher errors with "bad" samples against the reference curve.
- **threshold:** Provide a percentile value to determine the threshold value on the procrustes error distribution.
- **normalize:** Option to normalize traces with respect to themselves before fitting. Useful when traces have highly variables ranges. For example, stock prices.
- **ci_width:** Confidence interval width parameter. Useful for approximating the location of the divergence or the anomaly on a trace.

The procrustes and trace normalizing functionality can be directly imported from the metric module-

```python
from trendalation.metrics import procrustes
trace1_transformed, trace2_transformed, disparity = procrustes(trace1, trace2)
```

## Help and Support

In order to report bugfixes and new feature requests, simply create a new issue on the repository.
The issues will be reviewed by the authors on a regular basis & you're welcome to work on any open issues.

## Contribution

This project is a community effort, and everyone is welcome to contribute.
Feel free to work on any open issues and setup PRs.
