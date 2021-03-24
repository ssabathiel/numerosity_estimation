###############################
## CREATE CLASS: DATA-SET THAT CONTAINS:
##     IMG, IMG_FLATTEN, N_OBJ, N_OBJ_ONEHOT, TOTAL_AREAS,
##########################################################

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
import matplotlib.pyplot as plt


class SquareDataSet():
  
    def __init__(self, img_flatten, n_obj_onehot, total_areas=None):
        self.img_flatten = img_flatten
        self.n_obj_onehot = n_obj_onehot        
        self.img = np.reshape(img_flatten, (-1,img_size,img_size))
        self.n_obj = np.argmax(n_obj_onehot,axis=1)+1        
        self.hull = self.get_convex_hull()
        
        if(total_areas is None):
          self.total_areas = self.get_total_areas()
        else:
          self.total_areas = total_areas
                
    def subset_conditioned(self, area=-1, n_obj=-1, hull=-1):
        
        subset_ = SquareDataSet(self.img_flatten, self.n_obj_onehot,self.total_areas)
        
        if(area>=0):
          area_indices = np.where(subset_.total_areas == area)[0]
          subset_ = subset_.subset(subset_, area_indices)
        
        if(n_obj>=0):
          n_obj_indices = np.where(subset_.n_obj == n_obj)[0]
          subset_ = subset_.subset(subset_, n_obj_indices)
        
        if(hull>=0):
          n_hull_indices = np.where(subset_.hull == hull)[0]
          subset_ = subset_.subset(subset_, n_hull_indices)
        
        return subset_
          
    def subset(self, set_, indices):
        img_flatten = set_.img_flatten[indices]
        n_obj_onehot = set_.n_obj_onehot[indices]
        total_areas = set_.total_areas[indices]
        
        img = set_.img[indices]
        n_obj = set_.n_obj[indices]
        
        return SquareDataSet(img_flatten, n_obj_onehot,total_areas)
      
    def get_convex_hull(self):
            
        image = self.img[0]
        chull = convex_hull_image(image)
        binary_hull = chull.copy().astype(int)
        hull_scalar = np.sum(binary_hull)
        hull = hull_scalar
        
        for i in range(1,self.n_obj.size ):
            
            image = self.img[i]
            chull = convex_hull_image(image)
            binary_hull = chull.copy().astype(int)
            hull_scalar = np.sum(binary_hull)
            
            hull = np.vstack( (hull, hull_scalar) )
            
        return hull
      
    def get_total_areas(self):
       
        total_areas = np.sum(self.img[0]/255)

        for i in range(1,self.n_obj.size ):

            total_area = np.sum(self.img[i]/255)
            total_areas = np.vstack( (total_areas, total_area) )

        return total_areas
      

    def show_images(self,listy, hull=False ):
        #from https://scikit-image.org/docs/dev/auto_examples/edges/plot_convex_hull.html
        
        if(type(listy) == int):
          listy = range(listy)
        
        for i in listy:
                       
            if(hull):
                chull = convex_hull_image(self.img[i])
                chull_diff = img_as_float(chull.copy())
                img = self.img[i]/255
                img = img.astype(bool)
                chull_diff[img.reshape(28,28)] = 2

                fig, ax = plt.subplots()
                ax.imshow(chull_diff, cmap=plt.cm.gray)
                ax.set_title('Image and convex hull')

                plt.tight_layout()
                plt.show()
            else:
                img = self.img[i]/255
                img = img.astype(bool)

                fig, ax = plt.subplots()
                ax.imshow(img, cmap=plt.cm.gray)
                ax.set_title('Image')

                plt.tight_layout()
                plt.show()



###################################
## CREATE IMAGES WITH RANDOMLY                   NEWWWWWWWW
##  POSITIONED SQUARES WITH VARIED SIZE          NEWWWWWWWW: 1) possible individual areas: 1²,3²,5² --> 1²,2²,3²,4²,5²  2) control for fixed area, fixed individual area, fixed accumulative area, 
#################################
# Good source about pixel manipulation: http://pythoninformer.com/python-libraries/numpy/numpy-and-images/

import numpy as np
import scipy.misc as smp
import random
from PIL import Image
from IPython.display import display
from IPython.display import HTML
style = "<style>svg{width:500% !important;height:500% !important;}</style>"


max_objects_ = 10
img_size = 28

area_chunk = 5

def myround(x, base=5):
    return base * round(x/base)

###################################
## Function that creates N (= argument of the function) Squares
##  randomly positioned with varying size
##  AND corresponding one-hot-encoded number of objects .
########
def Create_N_Sqaures(n_objects, fixed_area=False, area_=1, rectangle=False, fixed_total_area=-1):

    
    min_side_length = 1
    max_side_length = 5
    
    data = np.zeros((img_size, img_size), dtype=np.uint8)                                          
    total_area = 0
    max_objects = max_objects_

    if(fixed_total_area>0):
        side_lengths_a, side_lengths_b = get_n_square_sides_for_fixed_area(n_objects, fixed_total_area)

    for n in range(n_objects):
      
        new_attempt_all_squares = True
        
        while(new_attempt_all_squares==True):
            breaky = 0
            max_put_try = 4000    #Number of attempts to place square correctly: without overlap, within the borders of the image.
            put_try=0
            
            side_length =  [random.randint(min_side_length,max_side_length) for i in range(2)]
            if(fixed_area):
                side_length = [area_, area_]
            if(fixed_total_area>0):
                side_length = [side_lengths_a[n], side_lengths_b[n]]
            

            # Now try to put square correctly into image for max_put_try attempts.
            # Strategy: choose random x and y to set left-lowest corner of square and check whethter it intersects with existing squares
            while (breaky == 0):
              
                dist_around_squ = 2
                dist_to_squ = dist_around_squ
                
                rand_pixel_1 = random.randint(dist_around_squ,img_size-side_length[0]-dist_around_squ)
                rand_pixel_2 = random.randint(dist_around_squ, img_size-side_length[1]-dist_around_squ)

                
                # Check if object does not intersect with other
                breaky = 1
                for i in range(-dist_to_squ, side_length[0] + dist_to_squ):
                    for j in range(-dist_to_squ, side_length[1] + dist_to_squ):
                          if(data[rand_pixel_1 + i, rand_pixel_2 + j] == 255):
                              breaky=0

                put_try += 1
                
                # If tried max_put_try/20 time with the current size create new size of square (gets chance to be smaller and fit now)
                if(put_try%(max_put_try/20)==0 ): 
                    side_length =  [random.randint(min_side_length,max_side_length) for i in range(2)]
                    if(fixed_area):
                        side_length = [area_, area_]
                
                if(put_try >= max_put_try):
                    breaky = 1
                    print("ATTENTION: OBJECTS COULD NOT FIT INTO WINDOW. CHOOSE DIFFERENT SIZE FOR WINDOW OR OBJECTS")
                    #exit()
                    new_attempt_all_squares = True
                else:
                    new_attempt_all_squares = False
                
        data[rand_pixel_1,rand_pixel_2] = 255

        for i in range(side_length[0]):
            for j in range(side_length[1] ):
                data[rand_pixel_1 + i, rand_pixel_2 + j] = 255

        square_area = side_length[0]*side_length[1]
        total_area +=square_area
    

    # Data 2D-image --> 1D-array ( for NN as Input )
    data_flatten = data.flatten()
    # Build one-hot array from number of objects
    n_obj_one_hot = np.zeros(max_objects)
    n_objects_m_1 = n_objects-1
    n_obj_one_hot[n_objects_m_1] = 1
    
    return data_flatten, n_obj_one_hot, total_area




###################################
## Function that creates N images with set of squares
##   AND corresponding one-hot-encoded number of objects.
######
def Create_N_Images(n_images, only_one_number=False, n_squares=5, fixed_area=False, area_=1, max_objects= max_objects_, rectangle=False, fixed_total_area=-1):

    areas = np.zeros((max_objects*25+1, max_objects+1))
    orig_areas = np.zeros((max_objects*25+1, max_objects+1))
  
    n_objects = random.randint(1, max_objects)
        
    if (only_one_number):
        n_objects = n_squares
    
    mult_img, mult_class, mult_total_areas = Create_N_Sqaures(n_objects, fixed_area, area_, rectangle, fixed_total_area)
    for i in range(n_images-1):
      
        n_objects = random.randint(1, max_objects)
        if (only_one_number):
            n_objects = n_squares

        data_flatten, n_obj_one_hot, total_area = Create_N_Sqaures(n_objects, fixed_area, area_, rectangle, fixed_total_area)
        mult_img = np.vstack([mult_img, data_flatten])
        mult_class = np.vstack((mult_class, n_obj_one_hot))
        mult_total_areas = np.vstack((mult_total_areas, total_area))
        
        areas[total_area][n_objects]+=1
        #orig_areas[orig_area][n_objects]+=1
        
    data_set = SquareDataSet(mult_img, mult_class, mult_total_areas)    
    
    return data_set




def get_n_square_sides_for_fixed_area(n_squares, total_area):
    totals = np.array([total_area])  # don't use Sum because sum is a reserved keyword and it's confusing
    rest = 1
    tried_too_many_times = True
    any_side_is_zero = True

    total_trial = 0

    while(tried_too_many_times or rest!=0 or any_side_is_zero):
        total_trial += 1
        #a = np.random.randint(2,5, size=(n_squares, 1))  # create random numbers
        a = np.random.random((n_squares, 1))  # create random numbers
        b = np.random.random((n_squares, 1))
        a = a/np.sqrt( np.sum(np.multiply(a, b), axis=0) * totals)
        b_ = b/np.sqrt( np.sum(np.multiply(a, b), axis=0) * totals)  # force them to sum to totals

        b = b_
        # Ignore the following if you don't need integers
        a = np.round(a)  # transform them into integers
        b = np.round(b)  # transform them into integers
        remainings = totals - np.sum(np.multiply(a, b), axis=0)  # check if there are corrections to be done
        for j, r in enumerate(remainings):  # implement the correction
            step = 1 if r > 0 else -1
            tried_too_many_times = False
            trial = 0
            while r != 0 and not tried_too_many_times:
                
                i = np.random.randint(n_squares)
                a_b = np.random.randint(2)
                if b[i,j] + step >= 0:
                    if(a_b==0):
                      b[i, j] += step
                      other = a[i, j]
                    else:
                      a[i, j] += step
                      other = b[i, j]
                    #r -= step
                    r = totals - np.sum(np.multiply(a, b), axis=0)
                    trial += 1
                    if(trial>1000):
                      tried_too_many_times=True
                      break
                      print('tried too many times')
        a = [int(a[i][0]) for i in range(n_squares)]
        b = [int(b[i][0]) for i in range(n_squares)]
        rest = r    
        any_side_is_zero = True
        if(0 not in a and 0 not in b):
          any_side_is_zero = False
          a = np.array(a)
          b = np.array(b)
        if(total_trial>200):
          print('TRY ANOTHER ALGORITHM')
          exit()
    
    return a, b

####################################
## DEMONSTRATION HOW TO USE DATA SET
####################################


####################
## 1) Create data set 
##################################
# --> will return an instance of the class SquareDataSet()
# 1st argument corresponds to number of pictures/samples
# only_one_number should be set true if you want all the samples to consist only pictures with n_squares squares. Fixed area should be set true if all the pictures have only squares with same side length = area_
#data_set = Create_N_Images(300, only_one_number=False, n_squares=5, fixed_area=False, area_=2)   # n_squares will only be considered if only_one_number==True,  area_ only if fixed_area==True

####################
## 2) Extract data from class  
##################################
# choose whatever you need to feed into your NN or use otherwise
# Each of the extracted will be an array of length n_samples
#img = data_set.img   
#img_flatten = data_set.img_flatten
#n_obj = data_set.n_obj
#n_obj_onehot = data_set.n_obj_onehot
#total_areas = data_set.total_areas
#hulls = data_set.hull

####################
## 3) Show images of the dataset  
##################################
# 1st argument: number of pictures that should be shown from the dataset. can also be given as list of indices 
# 2nd argument: if the pictures should be shown with or without the convex hull
#data_set.show_images(2, hull=True)
#data_set.show_images([3,8], hull=False)

####################
## 4) Get a subset of the data_set with certain constrains  
##################################
#subset_1 = data_set.subset_conditioned(area=25)
#subset_2 = data_set.subset_conditioned(n_obj=3)
#subset_3 = data_set.subset_conditioned(hull=255)
#subset_4 = data_set.subset_conditioned(area=27, n_obj=3)


## Remark
# The data set can be constrained at two points:
# 1. when creating the data set: on constant individual area and on the number of squares
# 2. after the data set has been created, one can take a subset of this: on the total area, on the convex hull and on the number of squares
# The reason the different constraints is that some of the parameters are easier controllable during the creation then afterwards or vice versa (such as the convex hull)
# Also if one wants to have a larger data set with certain constraints, it might be better to put them at creation time already, since the subset of a given data set with certain
# constraint (such as low areas), might be too small to do a proper analysis



  ######################################
## CREATE, SAVE AND LOAD DATA-SET
######################################
#                |  
#                | 
#                | 
#                | 
#                V  ######################################
## CREATE DATA SET
######################################
#from sklearn.model_selection import train_test_split

#data_set_train = Create_N_Images(21000, only_one_number=False, n_squares=5, fixed_area=False, area_=2)   
#data_set_test = Create_N_Images(9000, only_one_number=False, n_squares=5, fixed_area=False, area_=2)   

# Extract needed data   
#trX = data_set_train.img_flatten
#trY = data_set_train.n_obj_onehot
#teX = data_set_test.img_flatten
#teY = data_set_test.n_obj_onehot

#trX = data_set.img
#trY = data_set.n_obj

