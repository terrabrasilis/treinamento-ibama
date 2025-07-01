# -*- coding: utf-8 -*-
"""
Created on Fri May 16 06:55:14 2025

@author: cuel001
"""
import argparse
import requests
import logging
from datetime import datetime
from pathlib import Path
import os
import re
import glob
import numpy as np
from osgeo import gdal
import errno
import fnmatch
import sys
import imageio
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import RandomizedSearchCV
import rasterio

def save_as_geotiff(data, reference_tiff_path, output_tiff_path):
    with rasterio.open(reference_tiff_path) as ref:
        profile = ref.profile
        profile.update(dtype=rasterio.float32, count=data.shape[2])

        with rasterio.open(output_tiff_path, 'w', **profile) as dst:
            for t in range(data.shape[2]):
                dst.write(data[:, :, t].astype(rasterio.float32), t + 1)

def check_folder(folder_dir):
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get(endpoint, params=None):
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    print(f"error: {response.text}")
    return None

def download_asset(url, download_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"downloaded in {download_path}")
        return True
    print(f"error downloading file {url}: {response.text}")
    return False
    
  
def find_common_timesteps(data_path, feature_names):
    folders = os.listdir(data_path)
    
    
    feature_timesteps = {feature: [] for feature in feature_names}
    for f in folders:
        files = glob.glob(os.path.join(data_path,f, '*.tif'))
    
        for file in files:
            _, filename = os.path.split(file)
            feature, year, month, fortnight = re.match(r'([a-zA-Z0-9_]+)_?(\d{4})(\d{2})(\d{2}).tif', filename).groups()
            if feature in feature_names:
                feature_timesteps[feature].append((int(year), int(month), int(fortnight)))

    # Sort the time-steps for each feature
    for feature in feature_timesteps:
        feature_timesteps[feature].sort(key=lambda x: (x[0], x[1], x[2]))  # Sort by year, then month, then fortnight

    common_timesteps = set(feature_timesteps[feature_names[0]])
    for feature in feature_names[1:]:
        common_timesteps.intersection_update(feature_timesteps[feature])

    return sorted(list(common_timesteps))


def read_tiff(tiff_file):
    data = gdal.Open(tiff_file).ReadAsArray()
    return data


def load_geotiffs(data_path, feature_names, common_timesteps, quinzena=[], addmonth_info = False):
    feature_stacks = []

    for feature in feature_names:
        timestep_stacks = []

        for year, month, fortnight in common_timesteps:
            filename = f'{feature}{year}{month:02d}{fortnight:02d}.tif'
            file_path = os.path.join(data_path, f'{feature}', filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            timestep_stacks.append(file_path)

        #TODO: the [:98,:136] adjustment is provisional because some tif fliles have a diff size
        timestep_stacks = np.stack([read_tiff(x).astype('float32')[:98,:136] for x in timestep_stacks],axis=-1)
        
        feature_stacks.append(timestep_stacks)
        
    # add biweek info as raster
    if addmonth_info:
        cont = 0
        quin_stacks = []
        for year, month, fortnight in common_timesteps:
    
            quin_stacks.append(np.ones((timestep_stacks.shape[0],timestep_stacks.shape[1]))*quinzena[cont])
            cont+=1
    
        quin_stacks = np.stack(quin_stacks,axis=-1)
        
        feature_stacks.append(quin_stacks)

    # Assuming all features have the same spatial dimensions
    if len(feature_names) > 1 or addmonth_info:
        return np.stack(feature_stacks, axis=-1)
    else:
        return feature_stacks[0]
    
def find_files_with_extension(directory, extension):
    matching_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file matches the specified extension
            if fnmatch.fnmatch(file, '*' + extension):
                # Add the file to the list
                matching_files.append(os.path.join(root, file))

    return matching_files
    
def load_statics(data_path, feature_names, dynamic_array):
    feature_stacks = []
    
    all_files = find_files_with_extension(data_path, '.tif')

    for feature in feature_names:
        file_path = [x for x in all_files if feature in x]
        
        if len(file_path) < 1:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        static_var = read_tiff(file_path[0])
        if len(static_var.shape)>2:
            feature_stacks.append(np.rollaxis(static_var,0,3))
        else:
            feature_stacks.append(static_var[:, :, np.newaxis])
            
    #### static array to dynamic array 
    feature_stacks = np.concatenate(feature_stacks, axis=-1)
    
    if len(dynamic_array.shape) == 3:
        dynamic_array = dynamic_array[:, :, :, np.newaxis]
       
    feature_stacks = feature_stacks[:, :, np.newaxis, :]
    feature_stacks = np.repeat(feature_stacks, dynamic_array.shape[2], axis=2)
    
    dynamic_array = np.concatenate((dynamic_array,feature_stacks), axis=-1)
    
    return dynamic_array


def split_data(inputs, targets, train_range, val_range, test_range, common_timesteps):
    
    list_times = np.ones(6).astype('uint8')*(-1)
    cont_ind= 0

    for y,m,d in common_timesteps:
        strtime = f'{y}{m:02d}{d:02}'
        if (strtime >= train_range[0]) and (strtime <= train_range[1]):
            
            if list_times[0] == -1:
                list_times[0] = cont_ind
            else:
                list_times[1] = cont_ind

        elif (strtime >= val_range[0]) and (strtime <= val_range[1]):
            
            if list_times[2] == -1:
                list_times[2] = cont_ind-1
            else:
                list_times[3] = cont_ind

        elif (strtime >= test_range[0]) and (strtime <= test_range[1]):
            
            if list_times[4] == -1:
                list_times[4] = cont_ind-1
            else:
                list_times[5] = cont_ind+1
        
        cont_ind+=1
        
    
    train_inputs = inputs[:, :, list_times[0]:list_times[1]].copy().astype('float32')
    train_targets = targets[:, :, list_times[0]:list_times[1]].copy().astype('float32')

    val_inputs = inputs[:, :, list_times[2]:list_times[3]].copy().astype('float32')
    val_targets = targets[:, :, list_times[2]:list_times[3]].copy().astype('float32')

    test_inputs = inputs[:, :, list_times[4]:list_times[5]].copy().astype('float32')
    test_targets = targets[:, :, list_times[4]:list_times[5]].copy().astype('float32')

    return train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets


def reshape_and_clean_data(inputs, targets):
    #TODO: find better way to mask out the nan region
    mask = np.ones(targets[:,:,0].shape)
    mask[targets[:,:,0]>1000] = 0
    
    inputs, targets = inputs[mask==1], targets[mask==1]
    
    # Reshape the data to a 2D array (samples, features)
    num_samples = targets.shape[0] * targets.shape[1]
    try:
        num_features = inputs.shape[2]
    except:
        num_features = 1
        
    reshaped_inputs = inputs.reshape(num_samples, num_features)
    reshaped_targets = targets.reshape(num_samples)

    return reshaped_inputs, reshaped_targets


def main(args):
    
    check_folder(args.output_dir)

    
    if eval(args.verbose):
        log = logging.getLogger('')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # remove previous hanlder if any
        while log.handlers:
             log.handlers.clear()
        
        log.addHandler(ch)
        
        fh = logging.FileHandler(os.path.join(args.output_dir,'inference.log'))
        fh.setLevel(logging.INFO)
        ff = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ",
                                datefmt='%Y.%m.%d %H:%M:%S')
        fh.setFormatter(ff)
        log.addHandler(fh)

        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
    

    if args.download:
        logging.info("******************** Downloading data from server ******************** ")
        
        endpoint = f"{args.static_api_url}/collections/collection1"
        collection = get(endpoint=endpoint)
        assets = collection["item-assets"]
        for key in assets:
            print(f"\nasset: {assets[key]['title']} - {assets[key]['description']}")
    
        endpoint = f"{args.static_api_url}/search"
        
        beg = datetime(int(args.time_range[0][:4]), int(args.time_range[0][4:6]), int(args.time_range[0][6:]))
        end = datetime(int(args.time_range[1][:4]), int(args.time_range[1][4:6]), int(args.time_range[1][6:]))
        
        params = {
        "collections": ["collection1"],
        "datetime_range": beg.strftime("%Y-%m-%d")+'/'+end.strftime("%Y-%m-%d")  # correct format
        }
        
        items = get(endpoint=endpoint, params=params)
        items = items['features']
        
        root_download_dir = Path(f"{args.root_path}/{beg.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}")
    
        print("Downloading ......")
        for item in items:
            properties = item['properties']
            print(f"\nitem: {item['id']} - {properties['datetime']} - {list(item['assets'].keys())}")
            for name, values in item['assets'].items():
                url = values['href']
                print(f"\ndownloading {values['href']} ...")
            
                download_dir = root_download_dir / name
                download_dir.mkdir(exist_ok=True, parents=True)
                download_path = download_dir / Path(url).name
                download_asset(url=url, download_path=download_path)
                    
            logging.info("Downloading finished")
        
    else:
        beg = datetime(int(args.time_range[0][:4]), int(args.time_range[0][4:6]), int(args.time_range[0][6:]))
        end = datetime(int(args.time_range[1][:4]), int(args.time_range[1][4:6]), int(args.time_range[1][6:]))
        
        root_download_dir = Path(f"{args.root_path}/{beg.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}")
  
    logging.info("******************** Data preparation ******************** ")
    logging.info("Loading Dynamic Features")
    common_timesteps = find_common_timesteps(root_download_dir, args.dynamic)
    
    biweek_feature = [(m*2)-1 if q == 1 else m*2 for y, m, q in common_timesteps]
    
    stacked_array = load_geotiffs(root_download_dir, args.dynamic, common_timesteps, biweek_feature, args.add_biweek_info)

    ##### add  past quinzenas is defined
    if args.nb_biweek>1:
        logging.info("Adding past nb_biweek-1 biweeks ...")
        biweek_array = [stacked_array[:,:,args.nb_biweek-1:-1]]
        cont = 2
        for i in range(args.nb_biweek)[1:]:
            biweek_array.append(stacked_array[:,:,args.nb_biweek-1-i:-cont,0][...,np.newaxis])
            cont+=1
    
        stacked_array = np.concatenate(biweek_array, axis=-1)
    else:
        stacked_array = stacked_array[:,:,:-1]
        
    
    ##### add static data is defined
    if len(args.static) > 0:
        logging.info("Adding static data ...")
        stacked_array = load_statics(root_download_dir, args.static, stacked_array)
        

    logging.info("******************** Performing inference ...*******************")
    
    logging.info("Load model")
    model_path = os.path.join(args.output_dir, 'best_reg_model.json')
    best_model = xgb.XGBRegressor()
    best_model.load_model(model_path)
    
    test_inputs_reshaped, _ = reshape_and_clean_data(stacked_array, stacked_array[:,:,:,0])
    
    predictions_test = best_model.predict(test_inputs_reshaped)
    predictions_test[predictions_test < 0] = 0

    if args.apply_log:
        predictions_test = np.exp(predictions_test)-1
    
    # save geotiff file
    try:
        rows, cols, timesteps, bands = stacked_array.shape
    except:
        rows, cols, timesteps = stacked_array.shape
        
    # Initialize predictions array with the same shape, but only temporal dimension for predictions
    predictions_rg = np.full((rows, cols, timesteps), np.nan)
    
    reshaped_targets = stacked_array[:,:,:,0].reshape(-1)

    # Identify rows with non-NaN values for all bands
    #TODO: find better way to mask out nan values
    masktg = (reshaped_targets >= 0)&(reshaped_targets < 1000)

    # Initialize a flat predictions array to place predictions according to the mask
    predictions_flat = np.full(reshaped_targets.shape[0], np.nan)
    predictions_flat[masktg] = predictions_test

    # Reshape predictions back to the original shape
    predictions_rg = predictions_flat.reshape(rows, cols, timesteps)
    
    logging.info("Saving predictions and model ...")
    
    reference_tiff_path = find_files_with_extension(os.path.join(root_download_dir,args.target_name),'.tif')[0]
    
    save_as_geotiff(predictions_rg, reference_tiff_path, os.path.join(args.output_dir,'inference_reg.tif'))        
        
        
    logging.info("Processing finished ...") 
    

if __name__ == '__main__':
    """
    This script processes a GeoTIFF files for deforestation prediction. 
    It loads the files, converts it to a numpy array with size row x column x features x time step, 
    .The data is then split into 
    training, validation, and test sets based on specified timesteps.
    A XgBoost model is then trained
    """
    
    parser = argparse.ArgumentParser(description='Process GeoTIFF data for XGboost machine learning.')
    
    ############# PARAMETERS FOR DATA DOWNLOADING ################
    parser.add_argument('--static_api_url', type=str, default="https://terrabrasilis.dpi.inpe.br/stac-api/v1",
                        help='Data url')   
    parser.add_argument('--time_range', type=str, default=["20180101","20241216"],
                        help='Range time [begining, end]. Format yyyymmbb')    
    parser.add_argument('--download', type=bool, default=False,
                        help='If True download the dat, set to False if the data is already downloaded')
    
 
    
    ############# PARAMETERS FOR DATA PREPARATION AND MODEL ################
    parser.add_argument('--root_path', type=str, default='./data_test',
                        help='Path to the GeoTIFF file')    
    parser.add_argument('--output_dir', type=str, default='./exp/model_1',
                        help='Directory where the trained model will be saved')
    parser.add_argument('--dynamic', type=str, default=['Nuvem',  
                                                        'A7Q',  
                                                        'AcAr',  
                                                        'ArDS',  
                                                        'OcDS',  
                                                        'CtDS',  
                                                        'DeAr',  
                                                        'XQ',  
                                                        'PtDG',  
                                                        'Flor',  
                                                        'Pr',  
                                                        'NuAI',  
                                                        'DeAI',  
                                                        'PtEM'],
                        help='Names of the staick dynamic features to be loaded')
    parser.add_argument('--quartly_var',type=str, default=[],    #['XArDS','XDeDS']
                        help='Feature names for quarlty features')
    parser.add_argument('--static', type=str, default=['ACCESSCITY',  
                                                        'ACCESSBEEF',  
                                                        'ACCESSSOY',  
                                                        'ACCESSSOY25',  
                                                        'ACCESSWOOD',  
                                                        'CONN_MKT',  
                                                        'CONCITY10',  
                                                        'CONCITY100',  
                                                        'DVD',  
                                                        'DRYMONTHS',  
                                                        'EFAMS_IND',  
                                                        'EFAMS_UC',  
                                                        'EFAMS_TI',  
                                                        'EFAMS_FPND',  
                                                        'EFAMS_CAR',  
                                                        'EFAMS_ASS',  
                                                        'EFAMS_APA',  
                                                        'RODOFIC',  
                                                        'RODNOFIC',  
                                                        'DISTURB',  
                                                        'DISTRIOS',  
                                                        'DISTPORT'],
                    help='Names of the staick features to be loaded')
    
    parser.add_argument('--add_biweek_info', type=bool, default=True,
                        help='If True add biweek number as feature')
    parser.add_argument('--nb_biweek', type=int, default=3,
                        help='Total number of biweeks to consdier (current biweek + past biweek)')
    
    parser.add_argument('--test_timesteps', type=int, default=["20250101","20250416"],
                        help='Start and end date for test data')
    
    parser.add_argument('-v', '--verbose', action="store", dest='verbose', default = 'True',
                        help='Print log of processing')
    
    
    args = parser.parse_args()
    main(args)



    



