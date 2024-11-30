
from factor.factors import Factor
from factor.signal.Cross import Cross
from data.loader import Loader
from config.load import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import importlib, tqdm, factor, argparse, os
import loggings as log
from backtest.utils import FactorTest, index_contains
from flask import Flask, render_template_string, request, send_from_directory, render_template

app = Flask(__name__)

config = load_config()
TARGET = config.get("TARGET")
DELAY = config.get("DELAY")
HOURS = config.get("HOURS")

start_date = config.get("test_factor").get("start_date")
end_date = config.get("test_factor").get("end_date")

base_dir = os.path.abspath("factor/result")

@app.route('/get_distribution/<target>/<class_name>/<signal_name>')
def get_distribution(target, class_name, signal_name):
    directory = f"{base_dir}/{target}/{class_name}/{signal_name}"
    return send_from_directory(directory, "distribution.png")

@app.route('/get_ts_plot/<target>/<class_name>/<signal_name>')
def get_ts_plot(target, class_name, signal_name):
    directory = f"{base_dir}/{target}/{class_name}/{signal_name}"
    return send_from_directory(directory, "ts_plot.png")

@app.route('/')
def home():
    
    signal_name = request.args.get('signal_id')
    factor_cls = Factor.find_signal_class(signal_name)
    result_path = factor_cls.result_class_path(TARGET)
    
    report = pd.read_csv(os.path.join(result_path, "report.csv"))
    correlation = pd.read_csv(os.path.join(result_path, "corr.csv"))
    report_table = report.to_html(classes='data')
    corr_table = correlation.to_html(classes='data')
    
    plots = []
    for fctr in factor_cls.load_signals():
        fctr.load_target(TARGET)
        signal_name = str(fctr)
        plots.append(f'''
            <div class="plot-container">
                <img src="/get_distribution/{TARGET}/{fctr.__class__.__name__}/{signal_name}" class="plot-img">
                <img src="/get_ts_plot/{TARGET}/{fctr.__class__.__name__}/{signal_name}" class="plot-img">
            </div>
        ''')
        
    return render_template('report.html', report_table=report_table, corr_table=corr_table, plots=plots)


if __name__ == "__main__":
    print()
    app.run(debug=True)
