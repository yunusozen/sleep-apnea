# sleep-apnea

Convolutional neural network which takes as input 30 second single-channel EEG signals and produces as output a binary classification of whether an apnea event occurs in the signal. Includes implementation, validation, and example trained model.

### Requirements

* Python 3.5-3.8 installed (to use tensorflow)
* Training data from [Physionet](https://physionet.org/content/slpdb/1.0.0/). Store the data in the project folder data/mit-bih-polysomnographic-database-1.0.0.

### Setup

This project is best set up in a virtual environment to isolate package and language versions.

In the terminal, navigate into the project directory and create a new virtual environment using a compatible (3.5-3.8) version of Python: 

```
python3.8 -m venv env
```

Activate virtual environment:

```
source env/bin/activate
```

Install required packages:

```
python -m pip install -r requirements.txt
```


