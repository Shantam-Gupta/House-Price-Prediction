import json
import pickle
import os
import numpy as np
__locations = None
__data_columns = None
__model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())  # type: ignore
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns)) # type: ignore
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0: # type: ignore
        x[loc_index] = 1 # type: ignore

    return round(__model.predict([x])[0],2) # type: ignore

def get_location_names():
    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open(os.path.join(BASE_DIR, "artifacts", "columns.json"), "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open(os.path.join(BASE_DIR, "artifacts", "banglore_home_prices_model.pickle"), 'rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar',1000, 2, 2))
    print(get_estimated_price('Kalhalli',1000, 2, 2))
    print(get_estimated_price('Ejipura',1000, 2, 2))