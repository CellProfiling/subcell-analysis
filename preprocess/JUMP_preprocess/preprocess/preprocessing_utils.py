import numpy as np
from imageio import imread
from skimage import segmentation, measure
import cv2
import scipy.ndimage as ndi


def get_areas(cell_mask):
    return np.array([np.sum(cell_mask==i+1) for i in range(np.max(cell_mask))])

def get_bbox_areas(cell_mask):
   return [measure.regionprops(cell_mask)[i].area_bbox for i in range(np.max(cell_mask))]

def get_bbox(cell_mask):
    return [measure.regionprops(cell_mask)[i].bbox for i in range(np.max(cell_mask))]

def get_bbox_dims(cell_mask):
    sizes = []
    props = measure.regionprops(cell_mask)
    if len(props) != np.max(cell_mask):
        print(str(len(props)) + " != " + str(np.max(cell_mask)))
    for i in range(min(len(props), np.max(cell_mask))):
        x1, y1, x2, y2 = props[i].bbox
        sizes.append([x2-x1, y2-y1])
    return np.array(sizes)

def corners_to_center(l):
    x1, y1, x2, y2 = l
    return [x1+(x2-x1)//2, y1+(y2-y1)//2]

def center_crop(img, center, size):
    assert size % 2 == 0
    assert img.ndim == 3
    x_center, y_center = center
    crop = np.copy(img[x_center-size//2:x_center+size//2, y_center-size//2:y_center+size//2, :]) 
    return crop

def make_masks(cell_outline, nuc_outline):

    if np.max(cell_outline) !=1: cell_outline = cell_outline/np.max(cell_outline)
    if np.max(nuc_outline) !=1: nuc_outline = nuc_outline/np.max(nuc_outline)
    
    binarized_nuc = ndi.binary_fill_holes(nuc_outline).astype('int')

    binarized_nuc = binarized_nuc - nuc_outline #to reintroduce division between mitotic nuclei
    binarized_nuc = ndi.binary_fill_holes(binarized_nuc).astype('int') #to get rid of small loops in some segmentations


    nuc_labels, num_features = ndi.label(binarized_nuc) #will label 1-n
    nuc_labels[nuc_labels > 0] + 1 #want all labels to be >1, bc 1 will be background of watershed
    nuc_labels[cell_outline > 0] = 1 # background

    cell_mask = segmentation.watershed(cell_outline, nuc_labels)

    return cell_mask-1 #background=0, cells=1-to-n

'''
def sc_crop(img, cell_mask, cell_number, size, center="cell"):
    assert size % 2 == 0
    if center=="cell":
        x_center, y_center = np.round(measure.centroid(cell_mask==cell_number)).astype(np.int64)
    elif center=="bbox":
        x1, y1, x2, y2 = measure.regionprops(cell_mask)[cell_number-1].bbox
        x_center, y_center = x1+(x2-x1)//2, y1+(y2-y1)//2
    else:
        raise Exception("Wrong center type")
    crop = img[x_center-size//2:x_center+size//2, y_center-size//2:y_center+size//2]
    return crop
'''

def get_sc_crops(img, cell_mask, size, center="cell", reduce=True):
    assert size % 2 == 0
    
    #Get centers
    if center=="cell": #takes centroid of cell area
        centers = np.array([np.round(measure.centroid(cell_mask==i+1)) for i in range(np.max(cell_mask))])
    elif center=="bbox": #takes center of bounding box
        props = measure.regionprops(cell_mask)
        centers = [corners_to_center(props[i].bbox) for i in range(np.max(cell_mask))]
    else:
        raise Exception("Wrong center type")

    keep_idxs = np.arange(np.max(cell_mask))

    #Drop centers that are NaN
    nan_idxs = np.any(np.isnan(centers), axis=1)
    keep_idxs = np.array(list(set(keep_idxs) - set(nan_idxs)))
    centers = np.nan_to_num(centers).astype(np.int64)

    #Drop centers that are too close to border
    too_close_idxs = too_close_to_edge(centers, size, cell_mask)
    keep_idxs = np.array(list(set(keep_idxs) - set(too_close_idxs)))
    
    #Ignore centers whose crops have >50% area overlap 
    if reduce:
        keep_idxs = no_overlap(centers, size, dropped=set(too_close_idxs))

    if len(keep_idxs) > 0: #no centers left after filtering
        crops = np.array([crop_and_mask(img, centers[i], size, i+1) for i in keep_idxs])
        centers = centers[keep_idxs]
    else:
        crops = None
        centers = None

    return crops, centers

def crop_and_mask(img, center, size, cell_index):
    #assumes img has 5 channels + mask channel
    crop = center_crop(img, center, size) #get crop
    crop[:, :, -1] = crop[:, :, -1] == cell_index #reduce mask to only include cell i
    return crop

def pad(img, pad_len, type="constant"):
    assert img.ndim <= 3 and img.ndim >=2

    if img.ndim == 3:
        img = np.stack([pad(img[i], pad_len, type=type) for i in range(img.shape[0])])

    else:
        if type=="constant":
            img = cv2.copyMakeBorder(img, top=pad_len, bottom=pad_len, left=pad_len, right=pad_len, borderType=cv2.BORDER_CONSTANT, value=0)
        elif type=="reflect":
            img = cv2.copyMakeBorder(img, top=pad_len, bottom=pad_len, left=pad_len, right=pad_len, borderType=cv2.BORDER_REFLECT)
        elif type=="replicate":
            img = cv2.copyMakeBorder(img, top=pad_len, bottom=pad_len, left=pad_len, right=pad_len, borderType=cv2.BORDER_REPLICATE)
        else:
            raise Exception("Wrong padding type")
    return img

def dist_array(a):
    return np.array([np.abs(a-i) for i in a])
    #return np.abs(np.log(np.matmul(np.exp(a), np.exp(-a).T)))

def too_close_to_edge(centers, crop_size, cell_mask):
    '''Purpose: Determines which centers would have crop too close to edge of image, not enough padding for the crop
       Input: List of centers (shape = nx2), size of crop, cell_mask
       Output: List of indices of centers which can be dropped bc they are too close to edge 
    '''
    close_to_start = centers < crop_size/2 #too close to top or left
    close_to_end = (cell_mask.shape - centers) < crop_size/2 #to close to bottom or right
    too_close = np.any(np.logical_or(close_to_start,close_to_end), axis=1)

    return np.argwhere(too_close).flatten()

def no_overlap(centers, size, dropped=set(), threshold=0.5):
    '''Purpose: Want to delete centers for crops that overlap significantly. Uses greedy algorithm to determine which
        centers to drop.
       Input: List of centers (shape = nx2), size of crop, set of indices which we already want to drop
       Output: List of indices for a set of crop with acceptable overlap
    '''
    xs = centers[:, 0]
    ys = centers[:, 1]

    area_overlap = (size - dist_array(xs)).clip(0) * (size - dist_array(ys)).clip(0)
    assert np.all(area_overlap>=0)
    
    #droped idxs should have 0 overlap
    area_overlap[list(dropped), :]=0
    area_overlap[:, list(dropped)]=0 

    #crop doesn't overlap itself
    np.fill_diagonal(area_overlap, 0)
 
    sum_overlap = np.sum(area_overlap, axis=1)
    too_much_overlap = area_overlap > size*size*threshold
    keep_idxs = set(np.arange(0,centers.shape[0])) - dropped

    #Iteratively drop centers whose crop has maximal sum of area overlap with other crops
    while np.any(too_much_overlap[list(keep_idxs), :][:,list(keep_idxs)]):
        idx_to_drop = np.argmax(sum_overlap)
        keep_idxs.remove(idx_to_drop)

        #update area matrices now that we dropped a center
        area_overlap[idx_to_drop, :] = 0
        area_overlap[:, idx_to_drop] = 0
        sum_overlap = np.sum(area_overlap, axis=1)

    return list(keep_idxs)

def unfold_stack(crops):
    #Assumes crops is of shape (cells, height, width, channels)
    crops = np.transpose(crops, (3,1,2,0))
    crops = np.concatenate(crops, axis=1)
    crops = np.transpose(crops, (2, 0, 1))
    return crops
