import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import norm, skew #for some statistics

#libreria de garmin para leer archivos .fit
from garmin_fit_sdk import Decoder, Stream, Profile

stream = Stream.from_file(r"C:\Users\nicol\PycharmProjects\projects_ds\advanced_analytics\10397704682_ACTIVITY.fit")
decoder = Decoder(stream)
messages, errors = decoder.read()

print(errors)
print(messages)

record_fields = set()
def mesg_listener(mesg_num, message):
    if mesg_num == Profile['mesg_num']['RECORD']:
        for field in message:
            record_fields.add(field)

messages, errors = decoder.read(mesg_listener = mesg_listener)

if len(errors) > 0:
    print(f"Something went wrong decoding the file: {errors}")
    return

print(record_fields)
























































