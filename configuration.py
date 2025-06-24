import torch
import os

# This file configures global variables. Be careful when changing these!

dir_path = os.path.dirname(os.path.realpath(__file__))  # path of configuration file

data_path = dir_path + r"/data/market_data_csv.csv"
output_path = dir_path + r"/output/"

dpi_display = 100  # dpi for plotting to console
dpi_store = 200  # dpi for storing the plots

fig_size = (16, 9)  # measured in inch
font_size = 15
font_size_title = 20

date_column = 'Date'
target_column = '_MKT'
target_name = 'S&P 500'

time_step_size = 7  # measured in the unit of the time_step_unit variable
time_step_unit = 'days'  # this uses the pandas time units

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # very slow if no GPU is available

test_split = 0.2  # 20%
validation_split = 0.2  # 20%

batch_size = 250  # can be adjusted, but should not be too small for good performance

random_seed = 1234
