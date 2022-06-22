import glob
import numpy as np
import torch

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
import sys


import shutil
import custom_logger

import helper
from agx_update import agx_update
from slack_update import post_message

from scipy.stats import circmean

diffraction_downscaling = 1 

# Basic paths
base_path = "/lcrc/project/AICDI/S26_beamtime"
out_path = f"{base_path}/18June2022"
search_path = base_path + "/training_new"
overall_path = out_path + f"/overall/downscaled_{diffraction_downscaling}"
initial_training_data_path = out_path + "/initial"
best_model_params_path = overall_path + f"_best_model.pth"

# Default datasets for prediction (that the training does not access)
default_test_scan_indices = ['scan678.npz']#'scan571.npz', 'scan572.npz']
default_test_datasets = [f'{base_path}/Training2/{scan}' for scan in default_test_scan_indices]
default_test_slice_nums = [200, 500] # this is random


# Basic training parameters
EPOCHS = 50
NGPUS = torch.cuda.device_count()
BATCH_SIZE = NGPUS * 64
LR = 1e-3 


def setupOutputDirectoryAndLogging():
    outer_train_iterations = len(glob.glob(f'{out_path}/downscaled_{diffraction_downscaling}_iteration_*'))
    outer_train_iteration_this = outer_train_iterations + 1
    iteration_out_path = f'{out_path}/downscaled_{diffraction_downscaling}_iteration_{outer_train_iteration_this}'

    if (not os.path.isdir(iteration_out_path)):
        os.mkdir(iteration_out_path)


    custom_logger.setupLogging(iteration_out_path)
    return iteration_out_path

def printBasicParams():
    print("base path", base_path)
    print("out path", out_path)
    print("search path", search_path)
    print("overall results path", overall_path)
    print("Default testing datasets (tested after each training phase)", default_test_scan_indices)
    print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)
    

def searchAndCheckDatafiles():
    datafiles_all = glob.glob(search_path + "/*.npz")
    
    # Read list of datafiles already incorporated into training
    if os.path.isfile(f'{overall_path}/data_incorporated_list.txt'):
        with open(f'{overall_path}/data_incorporated_list.txt', 'r') as f:
            datafiles_incorporated = f.read().splitlines()
    else:
        datafiles_incorporated = []
    
    if len(datafiles_incorporated) > 0:
        print("Datafiles already included in model:")
        for df in datafiles_incorporated:
            print(df)
    
    datafiles_new = []
    for df in datafiles_all:
        if df not in datafiles_incorporated:
            continue_without_adding = False
            # Excluding default test scans from training
            if len(default_test_scan_indices) > 0:
                for scan in default_test_scan_indices:
                    if scan in df:
                        continue_without_adding = True
            
            if not continue_without_adding:
                datafiles_new.append(df)
    if len(datafiles_new) == 0:
        print("No new data detected.")
        exit(0)
    else:
        print("Datafiles not previously included:")
        for df in datafiles_new:
            print(df)
    return datafiles_incorporated, datafiles_new

def testPredictionQualityForNewDatafiles(datafiles_new):
    recon_model = helper.ReconModel()
    tester = helper.Tester(model=recon_model, batch_size=BATCH_SIZE, model_params_path=best_model_params_path)
    for df in datafiles_new:
        with np.load(df) as f:
            X_test = np.array(f['reciprocal'])
            positions_test = f['position']
            
            realspace = np.array(f['real'])
            phases = np.angle(realspace)
            phase_mean = circmean(phases, low=-np.pi, high=np.pi)
            Y_test = realspace * np.exp(-1j * phase_mean) 
            
        print('Predicting for data in ', df)

        fname_prefix = df.split('/')[-1].removesuffix('.npz')
        tester.setTestData(X_test)


        amps_eval, phs_eval = tester.predictTestData(npz_save_path=iteration_out_path + '/preds_' + fname_prefix + '.npz')

        Y_mag_test = np.abs(Y_test)
        Y_ph_test = np.angle(Y_test)

        tester.calcErrors(Y_mag_test, Y_ph_test, npz_save_path=iteration_out_path + '/errs_' + fname_prefix + '.npz')

        n_plot = 5
        selected = np.random.randint(X_test.shape[0], size=5)
        helper.plot_test_data(X_test[selected], Y_mag_test[selected], Y_ph_test[selected], amps_eval[selected], phs_eval[selected],
                             save_fname=iteration_out_path + '/test_imgs_' + fname_prefix + '.png', show_fig=False)
        print()
    
    
    
    
def trainWithAdditionalData(datafiles_incorporated: list, datafiles_new: list, iteration_out_path: str, load_model_path: str=None):
    print("Combining the training and test data for new training session.")
    datafiles_train = datafiles_incorporated + datafiles_new
    
    X_train = []
    Y_train = []
    positions_train = []
    for df in datafiles_train:
        with np.load(df) as f:

            X_train.append(np.array(f['reciprocal']))
            positions_train.append(f['position'])
            
            realspace = np.array(f['real'])
            phases = np.angle(realspace)
            phase_mean = circmean(phases, low=-np.pi, high=np.pi)
            Y_train.append(realspace * np.exp(-1j * phase_mean))
            
    print("Shape of new training data is", np.shape(X_train))

    shape12 = np.array(X_train).shape[-2:]
    X_train = np.reshape(X_train, [-1, *shape12])
    Y_train = np.reshape(Y_train, [-1, *shape12])
    Y_mag_train = np.abs(Y_train)
    Y_ph_train = np.angle(Y_train)


    print("After concatenating, shape of new training data is", np.shape(X_train))
        
    print("Before downscaling, max of X_train is", np.max(X_train))
    X_train = np.floor(X_train / diffraction_downscaling)
    print("After downscaling, max of X_train is", np.max(X_train))
    
    # The actual training part

    print("Creating the training model...")
    recon_model = helper.ReconModel()
    if load_model_path is not None:
        print("Loading previous best model to initialize the training model.")
        recon_model.load_state_dict(torch.load(best_model_params_path))

    print("Initializing the training procedure...")
    trainer = helper.Trainer(recon_model, batch_size=BATCH_SIZE, output_path=iteration_out_path, output_suffix='')
    print("Setting training data...")
    trainer.setTrainingData(X_train, Y_mag_train, Y_ph_train)
    print("Setting optimization parameters...")
    trainer.setOptimizationParams()
    trainer.initModel()
    
    trainer.run(EPOCHS)
    
    trainer.plotLearningRate(save_fname=iteration_out_path + '/learning_rate.png', show_fig=False)
    helper.plot_metrics(trainer.metrics, save_fname=iteration_out_path + '/metrics.png', show_fig=False)
    
    return datafiles_train

def updateIncorporatedDataList(datafiles_train):
    with open(f'{overall_path}/data_incorporated_list.txt', 'w') as f:
        for df in datafiles_train:
            print(df, file=f)
    
    
def runDefaultTests(iteration_out_path):
    recon_model = helper.ReconModel()
    tester = helper.Tester(model=recon_model, batch_size=BATCH_SIZE, model_params_path=iteration_out_path + '/best_model.pth')
    default_mean_pred_errors = []
    for df in default_test_datasets:
        with np.load(df) as f:
            X_test = np.array(f['reciprocal'])
            positions_test = f['position']
            
            realspace = np.array(f['real'])
            phases = np.angle(realspace)
            phase_mean = circmean(phases, low=-np.pi, high=np.pi)
            Y_test = realspace * np.exp(-1j * phase_mean) 
        
        print("Before downscaling, max of X_test is", np.max(X_test))
        X_test = np.floor(X_test / diffraction_downscaling)
        print("After downscaling, max of X_test is", np.max(X_test))
        
        print('Predicting for data in ', df)
        
        fname_prefix = df.split('/')[-1].removesuffix('.npz')
        tester.setTestData(X_test)


        amps_eval, phs_eval = tester.predictTestData(npz_save_path=iteration_out_path + '/preds_' + fname_prefix + '.npz')

        Y_mag_test = np.abs(Y_test)
        Y_ph_test = np.angle(Y_test)

        errors = tester.calcErrors(Y_mag_test, Y_ph_test, npz_save_path=iteration_out_path + '/errs_' + fname_prefix + '.npz')

        selected = default_test_slice_nums
        helper.plot_test_data(X_test[selected], Y_mag_test[selected], Y_ph_test[selected], amps_eval[selected], phs_eval[selected],
                             save_fname=iteration_out_path + '/test_imgs_' + fname_prefix + '.png', show_fig=False)
        print()
        
        default_mean_pred_errors.append(np.mean(errors, axis=0))
    return default_mean_pred_errors
    
    
    


if __name__ == '__main__':
    
    iteration_out_path = setupOutputDirectoryAndLogging()
    printBasicParams()
    
    datafiles_incorporated, datafiles_new = searchAndCheckDatafiles()
    
    print("Posting message to slack")
    post_message(f"PtychoNN: Starting new training run for diffraction data downscaled by a factor of {diffraction_downscaling}.")
    
    if os.path.isfile(best_model_params_path):  
        #default_mean_pred_errors = testPredictionQualityForNewDatafiles(datafiles_new)
        datafiles_train = trainWithAdditionalData(datafiles_incorporated, datafiles_new, iteration_out_path, load_model_path=best_model_params_path)
    else:
        datafiles_train = trainWithAdditionalData(datafiles_incorporated, datafiles_new, iteration_out_path, load_model_path=None)
    
    runDefaultTests(iteration_out_path)
    
    updateIncorporatedDataList(datafiles_train)
    
    print("Copying new best model to ", overall_path)
    shutil.copy(iteration_out_path + '/best_model.pth', best_model_params_path)
    
    #print("Updating agx")
    #agx_update(best_model_params_path)
    
    print("Posting message to slack")
    #post_message("PtychoNN: Pushed new model to AGX.")
    post_message("PtychoNN: Completed training run for diffraction data downscaled by a factor of {diffraction_downscaling}.")
    