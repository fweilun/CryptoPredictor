run `pip install -r requirements.txt` to install all dependencies

# How to Create a signal
The name of signal file, cfg file and class should be the same.
ex.
- python file: Cross.py
- json file: Cross.json
- class: Cross


# How to backtest a signal
1. clone the file to local workspace.
2. add signal file to signal folder.
3. add config file to cfg folder.
4. cd to base directory (CryptoPredictor).
5. run python -m backtest.test_factor $(signal_name)

ex. python -m backtest.test_factor weilun01

# Show all signal results

python -m backtest.test_factor all

# Show factor model backtest result

python -m backtest.factor_model




