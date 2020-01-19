import random
import glob
import pickle
from datetime import datetime

MODEL_FOLDER = 'models/'

def load_latest_model(network, model_name, folder):
    models = glob.glob(folder + '*.h5')

    # Load the latest model to use for self play
    if len(models) > 0:
        models.sort()
        name = models[len(models) - 1]
        if model_name != name:
            network.load_model(name)
        return name
    else:
        network.create_network(10)
        return None

def load_data_set(path):
    with open(path, 'rb') as filehandle:    
        data = pickle.load(filehandle)
    return data

def get_random_samples(data, num_samples):
    samples = []
    indexes = random.sample(range(len(data)), num_samples) if num_samples else range(len(data))
    for i in indexes:
        samples.append(data[i])
    return samples

def save_best_model(contender_network):
    models = glob.glob(MODEL_FOLDER + '*.h5')
    models.sort()
    if len(models) == 0:
        name = str(datetime.now()).replace(' ', '-') + '_model_0.h5'
    else:
        latest_model = models[len(models) - 1]    
        version = int(latest_model[latest_model.find('model_') + 6: latest_model.find('.h5')]) + 1
        name = str(datetime.now()).replace(' ', '-') + f'_model_{version}.h5'
    print(name)
    contender_network.save_model(MODEL_FOLDER + name)