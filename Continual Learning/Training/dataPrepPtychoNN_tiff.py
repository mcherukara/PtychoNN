import numpy as np 
from scipy import interpolate
import sys
import os
import fabio
from datetime import datetime

import time
from matplotlib import pyplot, colors
import glob 
from PIL import Image 
import hdf5plugin
import h5py

## for denoising 
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_nl_means)

def diff_data(reciprocalpath, scannum):
    
    dir_name = reciprocal_path+str(scannum)+"/"+str(scannum)+"/"
    list_of_files = sorted(filter(os.path.isfile,
                        glob.glob(dir_name + '*') ) )

    scan_arr = np.zeros((len(list_of_files), 512, 512))
    for i, file_path in enumerate(list_of_files):
        imarray = np.asarray(Image.open(file_path))
        imarray = imarray[2:-2,2:-2]
        scan_arr[i] = imarray[:,::-1]
    
    num_tiff = len(list_of_files)
    return scan_arr, num_tiff


def Prep_PtychoNN(recon_path, reciprocal_path, scannum, search_path):
    """
    recon_path: path to the Tike recon output 
    reciprocal: input diffraction patterns 
    scannum: scan ID as a command line argument 
    search_path: folder location to dump the preprocessed data for ptychoNN 
    Training code will monitor this folder for new files for training 

    """
    # reading the diff patterns from h5 file 
    reciprocal_path = reciprocal_path + str(scannum) 
    for df in glob.glob(reciprocal_path+'/scan*.h5'):
        with h5py.File(df) as f:
            data = f['entry/data/data'][()]
            data[data<0]=0
    
    
    ##post_message("preparing data for ptychoNN")

    angle_path = recon_path+ str(scannum) + '/scan-{0}_object_angle.tiff'.format(scannum)
    obj_ph =  np.asarray(Image.open(angle_path)) ## read from a tiff file 

    ## do image denoising 
    #obj_ph = denoise_nl_means(obj_ph, preserve_range=True)
    
    real_path = recon_path+ str(scannum) + '/scan-{0}_object_amp.tiff'.format(scannum)
    ampl = np.asarray(Image.open(real_path))
    


    pixelsize = 11.176e-9
    
    
    
    

    
    amp = ampl ## 
    
    pha = obj_ph


    pha_mean =  pha[int(pha.shape[0]/3.):int(pha.shape[0]/3.*2),int(pha.shape[1]/3.):int(pha.shape[1]/3.*2)].mean()
    pha -= pha_mean

    
    

   
    
    pos =  np.genfromtxt(reciprocal_path +'/positions.csv', delimiter=',') 
    
    

    x = np.arange(obj_ph.shape[1])*pixelsize
    y = np.arange(obj_ph.shape[0])*pixelsize
    x -= x.mean()
    y -= y.mean()

    fint_pha = interpolate.interp2d(x, y, pha, kind='cubic')
    fint_amp = interpolate.interp2d(x, y, amp, kind='cubic')

    
    real = np.zeros((pos.shape[0], 128, 128), dtype=np.complex64)
    xx = np.arange(128)*10e-9 # this is to predict on a 1 um area per point
    yy = np.arange(128)*10e-9
    xx -= xx.mean()
    yy -= yy.mean()

    for i in range(pos.shape[0]):
        real[i] = fint_amp(xx+pos[i,1], yy+pos[i,0])*np.exp(1j*fint_pha(xx+pos[i,1], yy+pos[i,0]))

    
    np.savez_compressed(search_path+"scan{0}.npz".format(scannum), real=real, reciprocal=data.astype('float32'), position=pos, pixelsize=pixelsize)


if __name__=="__main__":

    #bot_token = "xoxb-679835710832-2052497567909-oB0WeYpoEChiXF3FL0XYm1tb"
    #webclient = WebClient(token=bot_token)

    scanID = int(sys.argv[1])
    recon_path ="/grand/hp-ptycho/bicer/202206_run00_workflow-Tao/output/"
    search_path = "/grand/hp-ptycho/anakha/S26-beamtime/Training/"
    reciprocal_path = "/grand/hp-ptycho/bicer/202206_run00_workflow-Tao/input/"

    #data, num_tiff = diff_data(reciprocal_path, scanID)
    #for scanID in range(411, 422):
    Prep_PtychoNN(recon_path, reciprocal_path, scanID, search_path)


