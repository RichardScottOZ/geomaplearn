import numpy as np
# from numpy.random import choice
import keras
import imageio.v3 as iio
import tensorflow as tf

"""
This assumes a 2D image with 1 or 2 channels and 1 or 2 classes.
The first channel gives the geology in terms of relative age, and the second channel, 
if present gives the topographic elevation.

For a description of how to make a data generator, see here:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self, path, data_IDs, batch_size=32, dim=(256,256), 
             n_channels=2, shuffle=True, continuous_indices=False, 
             area=True, axis=True, batches_per_epoch=None, weights=None, 
             augment=False, fold_type='all', topo_scale=None):
    'Initialization'
    self.dim   = dim
    self.path = path
    self.batch_size = batch_size
    self.data_IDs = data_IDs
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.continuous_indices = continuous_indices
    self.area = area
    self.axis = axis
    if batches_per_epoch is None:
        self.batches_per_epoch = int(np.floor(len(self.list_IDs) / self.batch_size))
    else:
        self.batches_per_epoch = int(batches_per_epoch)
    self.weights = weights
    self.use_weights = not (self.weights is None)
    self.augment = augment
    self.fold_type = fold_type
    self.topo_scale = topo_scale
    self.epoch = -1 #on_epoch_end is called once before anything else is run, so at that point this will become 0.
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return self.batches_per_epoch

  def __getitem__(self, index):
    'Generates one batch of data'
    
    # If we are using continuous indices from one epoch to the next, add the number of batches from previous epochs.
    if self.continuous_indices:
        index = index+self.epoch*self.batches_per_epoch
    
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)
    
    #Do data augmentation.
    if self.augment:
        for i in range(self.batch_size):
            X[i,:,:,],Y[i,:,:] = augment(X[i,:,:,],Y[i,:,:])
    
    #Create a weights matrix if necessary.
    if self.use_weights:
        W = np.zeros((self.batch_size,self.dim[0],self.dim[1]),dtype=float)
        for i in range(self.batch_size):
            W[i,:,:] = add_sample_weights(Y[i,:,:],self.weights)
    
    #Return X, Y, and possibly W.
    if self.use_weights:
        return X, Y, W
    else:
        return X, Y

    

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.epoch += 1
    self.indexes = np.arange(len(self.data_IDs))
    if self.continuous_indices:
        if (self.epoch+1)*self.batches_per_epoch*self.batch_size > len(self.data_IDs): #We have reached the end of the dataset and need to start over.
            self.epoch = 0
            if self.shuffle == True:
              np.random.shuffle(self.indexes) #If using continuous indices, we only shuffle when (and if) we have gone through all indices once.
    else:
        if self.shuffle == True:
          np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    X = np.zeros((self.batch_size,self.dim[0],self.dim[1],self.n_channels),dtype=float)
    Y = np.zeros((self.batch_size,self.dim[0],self.dim[1]),dtype=int)
    for i, ID in enumerate(data_IDs_temp):
        filename = self.path+'/Model'+str(ID+1)
        unit = iio.imread(filename+'_unit.png').astype(float)
        if self.n_channels==1:
            image = np.expand_dims(unit,2) #expand_dims is needed to that the references to image[:,:,0] below don't cause problems.
        else:
            topo = iio.imread(filename+'_topo.png').astype(float)
            image = np.stack((unit,topo),axis=2)
        if self.n_channels>2:
            print('Error: Maximum of 2 data channels expected.')
        #Rescale the data to the range [0,1].
        #For the unit numbers, we first shift the numbers so that the non-Quaternary ones start at 1, while Quaternary is 0, and we remove any additional gaps in the numbering.
        #For topography, we set the minimum topography to 0 and the maximum to 1.
        units_unique = np.unique(image[:,:,0])
        if not np.any(units_unique==0):
            units_unique = np.concatenate([0],units_unique)
        for j in range(1,len(units_unique)):
            shift = units_unique[j] - units_unique[j-1] - 1
            if shift > 0: #There's a gap.
                image[:,:,0][image[:,:,0] >= units_unique[j]] -= shift
                units_unique[j:] -= shift
        image[:,:,0] = image[:,:,0]/image[:,:,0].max()
        if self.n_channels>1:
            # image[:,:,1] = (image[:,:,1]-image[:,:,1].min())/(image[:,:,1].max()-image[:,:,1].min())
            # image[:,:,1] = (image[:,:,1]-image[:,:,1].min())/self.topo_scale #Try consistent scaling of the topography.
            if self.topo_scale is None:
                image[:,:,1] = (image[:,:,1]-image[:,:,1].min())/(image[:,:,1].max()-image[:,:,1].min()) #Do we were doing before.
            else:
                image[:,:,1] = (image[:,:,1]-image[:,:,1].mean())/self.topo_scale
        X[i,:,:,:] = image
        if self.fold_type == 'all':
            suffix = 'sum'
        elif self.fold_type == 'anticlines':
            suffix = 'anticlines'
        elif self.fold_type == 'synclines':
            suffix = 'synclines'
        else:
            print('Error: Unrecognized fold type.')
        if self.axis and self.area:
            Y[i,:,:] = iio.imread(filename+'_axis_'+suffix+'.png') + iio.imread(filename+'_area_'+suffix+'.png')
        elif self.axis and not self.area:
            Y[i,:,:] = iio.imread(filename+'_axis_'+suffix+'.png')
        elif self.area and not self.axis:
            Y[i,:,:] = iio.imread(filename+'_area_'+suffix+'.png')
        else:
            print('Error: At least one of area or axis must be included as classes.')
    else:
        return X,Y

def add_sample_weights(label, weights):
  # Add weights based on the classification of a training image.
  # Weights should be a list with the number of elements equal to the number of classes in label.
  #This is based on the add_sample_weights function from this tutorial: 
  #    https://www.tensorflow.org/tutorials/images/segmentation#optional_imbalanced_classes_and_class_weights
  class_weights = tf.constant(weights)
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32)).numpy()

  return sample_weights

def augment(image,segmap):
    #Perform data augmentation by randomly cropping, flipping and rotating by 90 degrees.
    cropx = np.random.choice([-1,1,0])# 0=none, -1=left, 1=right
    cropy = np.random.choice([-1,1,0])# 0=none, -1=bottom, 1=top
    crop_px_x = int((np.random.rand()*0.3)*image.shape[0]) #Maximum of 30% crop.
    crop_px_y = int(np.random.rand()*0.3*image.shape[1])
    crop_mask = np.zeros(image[:,:,0].shape,dtype=bool)
    if cropx == -1:
        crop_mask[:crop_px_x,:] = True
    elif cropx == 1:
        crop_mask[-crop_px_x:,:] = True
    if cropy == -1:
        crop_mask[:,:crop_px_y] = True
    elif cropx == 1:
        crop_mask[:,-crop_px_y:] = True
    image[crop_mask,:] = 0
    segmap[crop_mask] = 0
    flip = np.random.choice([0,1,2]) #0 = none, 1 = left to right, 2 = top to bottom.
    if flip != 0:
        image = np.flip(image,axis=flip-1)
        segmap = np.flip(segmap,axis=flip-1)
    rot = np.random.choice([0,1,2,3]) #Number of 90 degree rotations to make.
    if rot != 0:
        image = np.rot90(image, k=rot, axes=(0, 1))
        segmap = np.rot90(segmap, k=rot, axes=(0, 1))
    return image,segmap