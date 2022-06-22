import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import skimage

def plot3(data: list, titles: list = [], save_fname: str =None):
    if(len(titles)<3):
        titles=["Plot1", "Plot2", "Plot3"]
    fig,ax = plt.subplots(1,3, figsize=(20,12))
    im=ax[0].imshow(data[0])
    ax[0].set_title(titles[0])
    ax[0].axis('off')
    plt.colorbar(im,ax=ax[0], fraction=0.046, pad=0.04)
    im=ax[1].imshow(data[1])
    ax[1].set_title(titles[1])
    ax[1].axis('off')
    plt.colorbar(im,ax=ax[1], fraction=0.046, pad=0.04)
    im=ax[2].imshow(data[2])
    ax[2].set_title(titles[2])
    ax[2].axis('off')
    plt.colorbar(im,ax=ax[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname)
    plt.show()
    

def getIntPositions(positions_all: np.ndarray,  # Set as (y,x). 
                    columnwise: bool = False, # whether the images are column-wise or row-wise. True corresponds to columnwise
                    pixel_size:float =8e-9, 
                    downsampling:float =1 
                   ):
    """ Use downsampling=1 if the positions and pixel size have already been downsampled. Use downsampling < 1 if we want to upsample
    the images before interpolation (to better account for subpixel shifts)."""
    
    raise NotImplementedError("I am not sure it is working correctly just yet.")
    if columnwise:
        pos_x = np.array(positions_all[:,0])
        pos_y = np.array(positions_all[:, 1])
    else:
        pos_x = np.array(positions_all[:,1])
        pos_y = np.array(positions_all[:,0])

    pos_row = (pos_x-np.min(pos_x)) / (pixel_size ) / downsampling
    pos_col = (pos_y-np.min(pos_y)) / (pixel_size ) / downsampling

    # integer position
    pos_int_row = pos_row.astype(np.int32)
    pos_int_col = pos_col.astype(np.int32)

    pos_subpixel_row = pos_row - pos_int_row
    pos_subpixel_col = pos_col - pos_int_col
    return pos_int_row, pos_int_col, pos_subpixel_row, pos_subpixel_col

def stitch(slices: np.ndarray, 
           positions_row: np.ndarray,
           positions_col: np.ndarray,
           columnwise: bool = False, # whether the images are column-wise or row-wise
           upsample_factor: int = 1, # Use > 1 if we want to 
          ):
    
    """The 'columnwise' part of this function and getIntPositions can be signified, but I can do that later.
    
    Use upsampling_factor > 1 if we want to upsample images before interpolation (to better account for subpixel shifts). 
    Assumes that the positions supplied are floats and accurately represent the current (not upsampled) scan position.
    """
    raise NotImplementedError("I am not sure it is working correctly just yet.")
    pos_int_row = (positions_row * upsample_factor).astype(np.int32)
    pos_int_col = (positions_col * upsample_factor).astype(np.int32)
    
    size = slices[0].shape[0]
    weights = None
    size_h = size // 2
    composite = np.zeros((np.max(pos_int_row) + size, np.max(pos_int_col) + size), slices.dtype)
    #print('Composite shape before trimming', composite.shape)
    ctr = np.zeros(composite.shape)
    if weights is None:
        weights = np.ones((np.array(slices[0].shape) * upsample_factor).astype('int32'), dtype='float32')

    for i in range(pos_int_row.shape[0]):
        
        slice_to_add = slices[i] if columnwise else slices[i].T
        if upsample_factor > 1:
            if slice_to_add.dtype in [np.complex64, np.complex128]:
                mag_slice_to_add = skimage.transform.rescale(np.abs(slice_to_add), upsample_factor, preserve_range=True)
                ph_slice_to_add = skimage.transform.rescale(np.angle(slice_to_add), upsample_factor, preserve_range=True)
                slice_to_add = mag_slice_to_add * np.exp(1j * ph_slice_to_add)
            else:
                slice_to_add = skimage.transform.rescale(slice_to_add, upsample_factor, preserve_range=True)
        
        composite[pos_int_row[i]: pos_int_row[i] + size, pos_int_col[i]: pos_int_col[i] + size] += slice_to_add * weights

        ctr[pos_int_row[i]:pos_int_row[i] + size, pos_int_col[i]:pos_int_col[i] + size] += weights#pb_weight
        

    composite = composite[size_h:-size_h,size_h:-size_h]
    ctr = ctr[size_h:-size_h, size_h:-size_h]

    composite /= (ctr + 1e-8)
    return composite

class ReconSmallPhaseModel(nn.Module):
    def __init__(self, nconv: int = 16):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv

        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8)
        )
        
        # amplitude model
        #self.decoder1 = nn.Sequential(
        #    *self.up_block(self.nconv * 8, self.nconv * 8),
        #    *self.up_block(self.nconv * 8, self.nconv * 4),
        #    *self.up_block(self.nconv * 4, self.nconv * 2),
        #    *self.up_block(self.nconv * 2, self.nconv * 1),
        #    nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)
        
        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
            nn.Tanh()
        )

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ]
        return block
    
    
    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]
        return block
        
    
    def forward(self,x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            #Restore -pi to pi range
            ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return ph
    
    
def plot_metrics(metrics: dict, save_fname: str = None, show_fig: bool = False):
    
    losses_arr = np.array(metrics['losses'])
    val_losses_arr = np.array(metrics['val_losses'])
    print("Shape of losses array is ", losses_arr.shape)
    fig, ax = plt.subplots(3,sharex=True, figsize=(15, 8))
    ax[0].plot(losses_arr[1:,0], 'C3o-', label = "Train")
    ax[0].plot(val_losses_arr[1:,0], 'C0o-', label = "Val")
    ax[0].set(ylabel='Loss')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend(loc='center right')
    ax[0].set_title('Total loss')
    
    #ax[1].plot(losses_arr[1:,1], 'C3o-', label = "Train Amp loss")
    #ax[1].plot(val_losses_arr[1:,1], 'C0o-', label = "Val Amp loss")
    #ax[1].set(ylabel='Loss')
    #ax[1].set_yscale('log')
    #ax[1].grid()
    #ax[1].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    #ax[1].set_title('Phase loss')
    
    ax[2].plot(losses_arr[1:,2], 'C3o-', label = "Train Ph loss")
    ax[2].plot(val_losses_arr[1:,2], 'C0o-', label = "Val Ph loss")
    ax[2].set(ylabel='Loss')
    ax[2].grid()
    #ax[2].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    ax[2].set_yscale('log')
    ax[2].set_title('Mag los')

    plt.tight_layout()
    plt.xlabel("Epochs")
    
    if save_fname is not None:
        plt.savefig(save_fname)
    if show_fig:
        plt.show()
    else:
        plt.close()
    

def plot_test_data(selected_diffs: np.ndarray,  selected_phs_true: np.ndarray, 
                   selected_phs_eval: np.ndarray, 
                   save_fname: str = None, show_fig: bool = True):
    
    n = selected_diffs_eval.shape[0]
    
    plt.viridis()
    
    f,ax=plt.subplots(7, n, figsize=(n * 4, 15))
    plt.gcf().text(0.02, 0.85, "Input", fontsize=20)
    plt.gcf().text(0.02, 0.72, "True I", fontsize=20)
    plt.gcf().text(0.02, 0.6, "Predicted I", fontsize=20)
    plt.gcf().text(0.02, 0.5, "Difference I", fontsize=20)
    plt.gcf().text(0.02, 0.4, "True Phi", fontsize=20)
    plt.gcf().text(0.02, 0.27, "Predicted Phi", fontsize=20)
    plt.gcf().text(0.02, 0.17, "Difference Phi", fontsize=20)

    for i in range(0,n):
        
        # display FT

        im=ax[0,i].imshow(np.log10(selected_diffs[i]))
        plt.colorbar(im, ax=ax[0,i], format='%.2f')
        ax[0,i].get_xaxis().set_visible(False)
        ax[0,i].get_yaxis().set_visible(False)


        # display predicted intens
        #im=ax[2,i].imshow(selected_amps_eval[i])
        #plt.colorbar(im, ax=ax[2,i], format='%.2f')
        #ax[2,i].get_xaxis().set_visible(False)
        #ax[2,i].get_yaxis().set_visible(False)

            # display original phase
        im=ax[4,i].imshow(selected_phs_true[i])
        plt.colorbar(im, ax=ax[4,i], format='%.2f')
        ax[4,i].get_xaxis().set_visible(False)
        ax[4,i].get_yaxis().set_visible(False)

        # display predicted phase
        im=ax[5,i].imshow(selected_phs_eval[i])
        plt.colorbar(im, ax=ax[5,i], format='%.2f')
        ax[5,i].get_xaxis().set_visible(False)
        ax[5,i].get_yaxis().set_visible(False)


        # Difference in phase
        im=ax[6,i].imshow(selected_phs_true[i] - selected_phs_eval[i])
        plt.colorbar(im, ax=ax[6,i], format='%.2f')
        ax[6,i].get_xaxis().set_visible(False)
        ax[6,i].get_yaxis().set_visible(False)
    
    if save_fname is not None:
        plt.savefig(save_fname)
    if show_fig:
        plt.show()
    else:
        plt.close()
    
class Trainer():
    def __init__(self, model: ReconSmallPhaseModel, batch_size: int, output_path: str, output_suffix: str):
        self.model = model
        self.batch_size = batch_size
        self.output_path = output_path
        self.output_suffix = output_suffix
        self.epoch = 0

    def setTrainingData(self, X_train_full: np.ndarray, Y_ph_train_full: np.ndarray,
                        valid_data_ratio: float = 0.1):
        
        self.H, self.W = X_train_full.shape[-2:]
        
        self.X_train_full = torch.tensor(X_train_full[:, None, ...].astype('float32'))
        self.Y_ph_train_full = torch.tensor(Y_ph_train_full[:, None, ...].astype('float32'))
        self.ntrain_full = self.X_train_full.shape[0]
        
        self.valid_data_ratio = valid_data_ratio
        self.nvalid = int(self.ntrain_full * self.valid_data_ratio)
        self.ntrain = self.ntrain_full - self.nvalid
                 
        
        self.train_data_full = TensorDataset(self.X_train_full, self.Y_ph_train_full)
        
        self.train_data, self.valid_data = torch.utils.data.random_split(self.train_data_full, [self.ntrain, self.nvalid])
        self.trainloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.validloader = DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        self.iters_per_epoch = int(np.floor((self.ntrain) / self.batch_size) + 1) #Final batch will be less than batch size
        
    
        
        
    
    def setOptimizationParams(self, epochs_per_half_cycle: int = 6, max_lr: float=1e-3, min_lr: float=1e-4):
        #Optimizer details
        
        self.epochs_per_half_cycle = epochs_per_half_cycle
        self.iters_per_half_cycle = epochs_per_half_cycle * self.iters_per_epoch #Paper recommends 2-10 number of iterations
        
        print("LR step size is:", self.iters_per_half_cycle, 
              "which is every %d epochs" %(self.iters_per_half_cycle / self.iters_per_epoch))
        
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        #criterion = lambda t1, t2: nn.L1Loss()
        self.criterion = self.customLoss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.max_lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, max_lr = self.max_lr, base_lr= self.min_lr,  
                                                           step_size_up = self.iters_per_half_cycle,
                                                           cycle_momentum=False, mode='triangular2')
    
    def testForwardSingleBatch(self):
        for ft_images, phs in self.trainloader:
            print("batch size:", ft_images.shape)
            ph_train = self.model(ft_images)
            print("Phase batch shape: ", ph_train.shape)
            print("Phase batch dtype", ph_train.dtype)

            loss_ph = self.criterion(ph_train, phs, self.ntrain)
            print("Phase loss", loss_ph)
            break
            
    def initModel(self, model_params_path: str = None):
        
        self.model_params_path = model_params_path
        if model_params_path is not None:
            self.model.load_state_dict(torch.load(self.model_params_path))
        summary(self.model, (1, 1, self.H, self.W), device="cpu")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model) #Default all devices

        self.model = self.model.to(self.device)
        
        print("Setting up mixed precision gradient calculation...")
        self.scaler = torch.cuda.amp.GradScaler()
        
        print("Setting up metrics...")
        self.metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
        print(self.metrics)
    
    
    def train(self):
        tot_loss = 0.0
        loss_ph = 0.0

        for i, (ft_images, phs) in enumerate(self.trainloader):
            ft_images = ft_images.to(self.device) #Move everything to device
            phs = phs.to(self.device)

            pred_phs = self.model(ft_images) #Forward pass

            #Compute losses
            loss_p = self.criterion(pred_phs, phs, self.ntrain) #Monitor phase loss but only within support (which may not be same as true amp)
            loss = loss_p #Use equiweighted amps and phase

            
            #Zero current grads and do backprop
            self.optimizer.zero_grad() 
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            tot_loss += loss.detach().item()

            loss_ph += loss_p.detach().item()

            #Update the LR according to the schedule -- CyclicLR updates each batch
            self.scheduler.step() 
            self.metrics['lrs'].append(self.scheduler.get_last_lr())
            self.scaler.update()


        #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        self.metrics['losses'].append([tot_loss, loss_ph]) 
    

    def validate(self):
        tot_val_loss = 0.0
        val_loss_ph = 0.0
        for j, (ft_images, phs) in enumerate(self.validloader):
            ft_images = ft_images.to(self.device)
            phs = phs.to(self.device)
            pred_phs = self.model(ft_images) #Forward pass


            val_loss_p = self.criterion(pred_phs,phs, self.nvalid)
            val_loss = val_loss_p

            #try complex valued diff
            #diff_real = pred_amps * torch.cos(pred_phs) - amps * torch.cos(phs)
            #diff_imag = pred_amps * torch.sin(pred_phs) - amps * torch.sin(phs)
            #val_loss = torch.mean(torch.abs(diff_real + diff_imag))

            tot_val_loss += val_loss.detach().item() 
            val_loss_ph += val_loss_p.detach().item() 

        self.metrics['val_losses'].append([tot_val_loss, val_loss_ph])


        self.saveMetrics(self.metrics, self.output_path, self.output_suffix)
        #Update saved model if val loss is lower

        if(tot_val_loss < self.metrics['best_val_loss']):
            print("Saving improved model after Val Loss improved from %.5f to %.5f" %(self.metrics['best_val_loss'],tot_val_loss))
            self.metrics['best_val_loss'] = tot_val_loss
            self.updateSavedModel(self.model, self.output_path, self.output_suffix)
            
    @staticmethod        
    def customLoss(t1, t2, scaling):
        return torch.sum(torch.mean(torch.abs(t1 - t2), axis=(-1, -2))) / scaling


    @staticmethod
    #Function to update saved model if validation loss is minimum
    def updateSavedModel(model: ReconSmallPhaseModel, path: str, output_suffix: str=''):
        if not os.path.isdir(path):
            os.mkdir(path)
        fname = path + '/best_model' + output_suffix + '.pth'
        print("Saving best model as ", fname)
        torch.save(model.module.state_dict(), fname) 
    
    @staticmethod
    def saveMetrics(metrics: dict, path: str, output_suffix: str=''):
        np.savez(path + '/metrics' + output_suffix + '.npz', **metrics)
        
        
    def run(self, epochs: int, output_frequency: int = 1 ):
        for epoch in range (epochs):
            
            #Set model to train mode
            self.model.train()
            
            #Training loop
            self.train()
            
            #Switch model to eval mode
            self.model.eval()
            
            #Validation loop
            self.validate()
            if epoch % output_frequency == 0:
                print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' 
                      %(epoch, self.metrics['losses'][-1][0], self.metrics['val_losses'][-1][0]))
                print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' 
                      %(epoch, self.metrics['losses'][-1][1], self.metrics['val_losses'][-1][1]))
                print('Epoch: %d | Ending LR: %.6f ' %(epoch, self.metrics['lrs'][-1][0]))
                print()
    
    
    def plotLearningRate(self, save_fname: str = None, show_fig: bool = True):
        batches = np.linspace(0, len(self.metrics['lrs']), len(self.metrics['lrs'])+1)
        epoch_list = batches / self.iters_per_epoch

        plt.plot(epoch_list[1:], self.metrics['lrs'], 'C3-')
        plt.grid()
        plt.ylabel("Learning rate")
        plt.xlabel("Epoch")
        
        plt.tight_layout()
        if save_fname is not None:
            plt.savefig(save_fname)
        if show_fig:
            plt.show()
        else:
            plt.close()
        
class Tester():
    def __init__(self, model: ReconSmallPhaseModel, batch_size: int, model_params_path: str):
        
        self.model = model
        self.batch_size = batch_size
        self.model_params_path = model_params_path
        
        self.model.load_state_dict(torch.load(self.model_params_path))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model) #Default all devices

        self.model = self.model.to(self.device)
        

    def setTestData(self, X_test: np.ndarray):
        self.X_test = torch.tensor(X_test[:,None,...].astype('float32'))
        self.test_data = TensorDataset(self.X_test)

        self.testloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

    
    def predictTestData(self, npz_save_path: str=None):
        self.model.eval()
        phs_eval = []
        for i, ft_images in enumerate(self.testloader):
            ft_images = ft_images[0].to(self.device)
            ph_eval = self.model(ft_images)
            for j in range(ft_images.shape[0]):
                phs_eval.append(ph_eval[j].detach().to("cpu").numpy())
        self.phs_eval = np.array(phs_eval).squeeze().astype('float32')
        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, ph=self.phs_eval)#mag=self.amps_eval, ph=self.phs_eval)
        #return self.amps_eval, self.phs_eval
        return self.phs_eval
    
    def calcErrors(self,  phs_true: np.ndarray, npz_save_path: str = None):
        from skimage.metrics import mean_squared_error as mse
        
        
        self.phs_true = phs_true
        self.errors = []
        for i, (p1, p2) in enumerate(zip(self.phs_eval, self.phs_true)):
            err2 = mse(p1, p2)
            self.errors.append([err2])
        
        self.errors = np.array(self.errors)
        print("Mean errors in phase")
        print(np.mean(self.errors, axis=0))
        
        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, phs_err=self.errors[:,0])
            
        return self.errors
        
            
        
        
    