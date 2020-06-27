import astroalign as aa
import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm, SqrtStretch
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils.psf import extract_stars
from photutils import EPSFBuilder,CircularAperture, find_peaks, DAOPhotPSFPhotometry

from skimage.transform import estimate_transform
from skimage.transform import matrix_transform

def data_correction(images,fdarkff,fdark,fff): 
    """Corrects images for dark, bias and flat field.
    
    Parameters
    ----------
    images : ndarray
        Raw images.
    fdarkff : string
        Location of the dark images for the flat field images.
    fdark : string
        Location of the dark images for the raw images.
    fff : string
        Location of the flat field images.
    Returns
    -------
    images: ndarray
        Corrected images
    """
    
    #Load dark images
    edark = os.listdir(fdark)
    dark = np.array([fits.getdata(fdark+'/'+edark[0], ext=0)])
    for k in range(1,len(edark)):
        dark = np.append(dark,[fits.getdata(fdark+'/'+edark[k], ext=0)],axis=0)
    
    #Create master dark
    master_dark = np.median(dark,axis=0)
    
    #Load flat field dark images
    edarkff = os.listdir(fdarkff)
    darkff = np.array([fits.getdata(fdarkff+'/'+edarkff[0], ext=0)])
    for k in range(1,len(edarkff)):
        darkff = np.append(darkff,[fits.getdata(fdarkff+'/'+edarkff[k], ext=0)],axis=0)   
    
    #Create master dark for flat field
    master_darkff = np.median(darkff,axis=0)
    
    #Load flat field images
    eff = os.listdir(fff)
    master_ff = np.array([master_darkff])
    for ef in eff:
        eff2 = os.listdir(fff+'/'+ef)
        ff_norm1 = fits.getdata(fff+'/'+ef+'/'+eff2[0], ext=0)-master_darkff
        ff_norm1 = ff_norm1/np.median(ff_norm1)
        ff = np.array([ff_norm1])
        for k in range(1,len(eff2)):
            ff_norm = fits.getdata(fff+'/'+ef+'/'+eff2[k], ext=0)-master_darkff
            ff_norm = ff_norm/np.median(ff_norm)
            ff = np.append(ff,[ff_norm],axis=0) 
        
        master_ff = np.append(master_ff,[np.median(ff,axis=0)],axis=0)
    
    #Create master flat field
    master_ff = np.median(master_ff[1:],axis=0)
    
    #Correct raw images
    for k in range(0,np.shape(images)[0]):
        images[k] = (images[k]-master_dark)/master_ff
    
    return images
   
def find_pinholes(fname,freference,sname,fdarkff,fdark,fff,files,size,threshold,fwhm,fitshape,sigma=2.,oversampling=4,maxiters=3):
    """Finds and fits pinhole positions with a ePSF in a FITS image. Then matches them to the reference positions.
    
    Parameters
    ----------
    fname : str
        Folder name of the input fits files.
    freference : str
        File name of the reference positions (txt file).
    sname : str
        Folder name of the returned found and matched pinhole positions (txt files).
    fdarkff : string
        Location of the dark images for the flat field images.
    fdark : string
        Location of the dark images for the raw images.
    fff : string
        Location of the flat field images.
    files : (1, 2)-shaped int array
        File range to create a median image
    size : int
        Rectangular size of the ePSF. Size must be an odd number.
    threshold : float
        The absolute image value above which to select sources.
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the Gaussian kernel in units of pixels.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be used to collect the data to do the fitting. 
        Can be an integer to be the same along both axes. E.g., 5 is the same as (5, 5), which means to fit only at the following 
        relative pixel positions: [-2, -1, 0, 1, 2]. Each element of fitshape must be an odd number.
    sigma : float
        Number of standard deviations used to perform sigma clip with a astropy.stats.SigmaClip object.
    oversampling : int or tuple of two int
        The oversampling factor(s) of the ePSF relative to the input stars along the x and y axes. 
        The oversampling can either be a single float or a tuple of two floats of the form (x_oversamp, y_oversamp). 
        If oversampling is a scalar then the oversampling will be the same for both the x and y axes.
    maxiters : int
        The maximum number of iterations to perform.
    Returns
    -------
    s_list : (N,2)-shaped array
        Found and matched positions of the pinholes.
    t_list : (N,2)-shaped array
        Matched reference grid positions.
    """
    
    #Load the sample of fits images
    entries = os.listdir(fname)
    
    data_col = np.array([fits.getdata(fname+'/'+entries[files[0]], ext=0)])
    for k in range(files[0]+1,files[1]+1):
        data_col = np.append(data_col,[fits.getdata(fname+'/'+entries[k], ext=0)],axis=0)
        
    #Data reduction: Darc current + Flatfield
    data_col = data_correction(images,fdarkff,fdark,fff)
        
    #Claculate median image
    data_full = np.median(data_col,axis=0)
    pos_full = np.array([[0,0]])
            
    data = data_full

    #Find peaks in data
    peaks_tbl = find_peaks(data, threshold=threshold) 
    peaks_tbl['peak_value'].info.format = '%.8g'

    #Load data around found peaks
    hsize = (size - 1) / 2
    x = peaks_tbl['x_peak']  
    y = peaks_tbl['y_peak']  
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
             (y > hsize) & (y < (data.shape[0] -1 - hsize)))  

    stars_tbl = Table()
    stars_tbl['x'] = x[mask]  
    stars_tbl['y'] = y[mask]  

    #Calculate mean, median, std
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=sigma)  
    data = data - median_val  

    #Find pinholes and create ePSF
    nddata = NDData(data=data)  

    stars = extract_stars(nddata, stars_tbl, size=size) 

    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
                               progress_bar=False)  
    epsf, fitted_stars = epsf_builder(stars)  
            
    #Use ePSF to find precise locations of pinholes
    daofind = DAOPhotPSFPhotometry(crit_separation=30, threshold=threshold, fwhm=fwhm, psf_model=epsf, 
                                   fitshape=fitshape,aperture_radius=12,niters=1)
                                   
    #Get positions
    sources = daofind(data)
            
    for col in sources.colnames:  
        sources[col].info.format = '%.8g'

    pos = np.transpose((sources['x_fit'], sources['y_fit']))
    pos_full = np.append(pos_full,pos,axis=0)
    
    pos_full = pos_full[1:]
    
    #Plot found pinholes
    apertures = CircularAperture(pos_full, r=10)

    norm = ImageNormalize(stretch=SqrtStretch())

    #Plot found pinholes
    fig, ax = plt.subplots()
    ax.set_title('Pinhole Positions')
    ax.set(xlabel='x [pixel]', ylabel='y [pixel]')
    ax.imshow(data_full, cmap='Greys', origin='lower', norm=norm)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    ax.legend(['#pinholes = '+str(len(pos_full[:,0]))]
                   ,loc='lower left',prop={'size': 12})
    plt.show()
    
    #Sort positions by matching with reference grid
    positions_sort = pos_full
    
    ref_positions = np.genfromtxt(freference,skip_header=0)
    
    transf, (s_list, t_list) = find_transform(positions_sort, ref_positions)
    
    text = np.array([s_list[:,0],s_list[:,1],t_list[:,0],t_list[:,1]])
    text_trans = np.zeros((len(s_list[:,0]),4))
    
    #Transpose text matrix
    for k in range(0,4):
        for l in range(0,len(s_list[:,0])):
            text_trans[l][k] = text[k][l]
    
    #Save data as txt file
    np.savetxt(sname+'.txt',text_trans,fmt='%1.9E',delimiter='\t',comments='',
               header='x-measured   y-measured   x-reference   y-reference')
    
    return s_list, t_list
    
MAX_CONTROL_POINTS = 2000
"""The maximum control points (stars) to use to build the invariants.
Default: 2000"""

PIXEL_TOL = 10
"""The pixel distance tolerance to assume two invariant points are the same.
Default: 10"""

MIN_MATCHES_FRACTION = 0.8
"""The minimum fraction of triangle matches to accept a transformation.
If the minimum fraction yields more than 10 triangles, 10 is used instead.
Default: 0.8
"""

NUM_NEAREST_NEIGHBORS = 5
"""
The number of nearest neighbors of a given star (including itself) to construct
the triangle invariants.
Default: 5
"""


def _invariantfeatures(x1, x2, x3):
    "Given 3 points x1, x2, x3, return the invariant features for the set."
    sides = _np.sort(
        [
            _np.linalg.norm(x1 - x2),
            _np.linalg.norm(x2 - x3),
            _np.linalg.norm(x1 - x3),
        ]
    )
    return [sides[2] / sides[1], sides[1] / sides[0]]


def _arrangetriplet(sources, vertex_indices):
    """Return vertex_indices ordered in an (a, b, c) form where:
  a is the vertex defined by L1 & L2
  b is the vertex defined by L2 & L3
  c is the vertex defined by L3 & L1
and L1 < L2 < L3 are the sides of the triangle defined by vertex_indices."""
    ind1, ind2, ind3 = vertex_indices
    x1, x2, x3 = sources[vertex_indices]

    side_ind = _np.array([(ind1, ind2), (ind2, ind3), (ind3, ind1)])
    side_lengths = list(map(_np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1)))
    l1_ind, l2_ind, l3_ind = _np.argsort(side_lengths)

    # the most common vertex in the list of vertices for two sides is the
    # point at which they meet.
    from collections import Counter

    count = Counter(side_ind[[l1_ind, l2_ind]].flatten())
    a = count.most_common(1)[0][0]
    count = Counter(side_ind[[l2_ind, l3_ind]].flatten())
    b = count.most_common(1)[0][0]
    count = Counter(side_ind[[l3_ind, l1_ind]].flatten())
    c = count.most_common(1)[0][0]

    return _np.array([a, b, c])


def _generate_invariants(sources):
    """Return an array of (unique) invariants derived from the array `sources`.
Return an array of the indices of `sources` that correspond to each invariant,
arranged as described in _arrangetriplet.
"""
    from scipy.spatial import KDTree
    from itertools import combinations
    from functools import partial

    arrange = partial(_arrangetriplet, sources=sources)

    inv = []
    triang_vrtx = []
    coordtree = KDTree(sources)
    # The number of nearest neighbors to request (to work with few sources)
    knn = min(len(sources), NUM_NEAREST_NEIGHBORS)
    for asrc in sources:
        __, indx = coordtree.query(asrc, knn)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        all_asterism_triang = [
            arrange(vertex_indices=list(cmb)) for cmb in combinations(indx, 3)
        ]
        triang_vrtx.extend(all_asterism_triang)

        inv.extend(
            [
                _invariantfeatures(*sources[triplet])
                for triplet in all_asterism_triang
            ]
        )

    # Remove here all possible duplicate triangles
    uniq_ind = [
        pos for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1 :]
    ]
    inv_uniq = _np.array(inv)[uniq_ind]
    triang_vrtx_uniq = _np.array(triang_vrtx)[uniq_ind]

    return inv_uniq, triang_vrtx_uniq


class _MatchTransform:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def fit(self, data):
        """
    Return the best 2D similarity transform from the points given in data.
    data: N sets of similar corresponding triangles.
        3 indices for a triangle in ref
        and the 3 indices for the corresponding triangle in target;
        arranged in a (N, 3, 2) array.
        """
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        approx_t = estimate_transform(
            "similarity", self.source[s], self.target[d]
        )
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(
            d1, d2
        )
        error = resid.max(axis=1)
        return error


def find_transform(source, target):
    """Estimate the transform between ``source`` and ``target``.
    Return a SimilarityTransform object ``T`` that maps pixel x, y indices from
    the source image s = (x, y) into the target (destination) image t = (x, y).
    T contains parameters of the tranformation: ``T.rotation``,
    ``T.translation``, ``T.scale``, ``T.params``.
    Args:
        source (array-like): Either a numpy array of the source image to be
            transformed or an interable of (x, y) coordinates of the target
            control points.
        target (array-like): Either a numpy array of the target (destination)
            image or an interable of (x, y) coordinates of the target
            control points.
    Returns:
        The transformation object and a tuple of corresponding star positions
        in source and target.::
            T, (source_pos_array, target_pos_array)
    Raises:
        TypeError: If input type of ``source`` or ``target`` is not supported.
        ValueError: If it cannot find more than 3 stars on any input.
    """
    from scipy.spatial import KDTree

    try:
        if len(source[0]) == 2:
            # Assume it's a list of (x, y) pairs
            source_controlp = _np.array(source)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            source_controlp = _find_sources(source)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError("Input type for source not supported.")

    try:
        if len(target[0]) == 2:
            # Assume it's a list of (x, y) pairs
            target_controlp = _np.array(target)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            target_controlp = _find_sources(target)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError("Input type for target not supported.")

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise ValueError(
            "Reference stars in source image are less than the "
            "minimum value (3)."
        )
    if len(target_controlp) < 3:
        raise ValueError(
            "Reference stars in target image are less than the "
            "minimum value (3)."
        )

    source_invariants, source_asterisms = _generate_invariants(source_controlp)
    source_invariant_tree = KDTree(source_invariants)

    target_invariants, target_asterisms = _generate_invariants(target_controlp)
    target_invariant_tree = KDTree(target_invariants)

    # r = 0.1 is the maximum search distance, 0.1 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches_list = source_invariant_tree.query_ball_tree(
        target_invariant_tree, r=0.1
    )

    # matches unravels the previous list of matches into pairs of source and
    # target control point matches.
    # matches is a (N, 3, 2) array. N sets of similar corresponding triangles.
    # 3 indices for a triangle in ref
    # and the 3 indices for the corresponding triangle in target;
    matches = []
    # t1 is an asterism in source, t2 in target
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(list(zip(t1, t2)))
    matches = _np.array(matches)

    inv_model = _MatchTransform(source_controlp, target_controlp)
    n_invariants = len(matches)
    max_iter = n_invariants
    # Set the minimum matches to be between 1 and 10 asterisms
    min_matches = max(1, min(10, int(n_invariants * MIN_MATCHES_FRACTION)))
    if (len(source_controlp) == 3 or len(target_controlp) == 3) and len(
        matches
    ) == 1:
        best_t = inv_model.fit(matches)
        inlier_ind = _np.arange(len(matches))  # All of the indices
    else:
        best_t, inlier_ind = _ransac(
            matches, inv_model, 1, max_iter, PIXEL_TOL, min_matches
        )
    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(tuple(pair) for pair in inl_arr)
    # In the next, multiple assignements to the same source point s are removed
    # We keep the pair (s, t) with the lowest reprojection error.
    inl_dict = {}
    for s_i, t_i in inl_unique:
        # calculate error
        s_vertex = source_controlp[s_i]
        t_vertex = target_controlp[t_i]
        s_vertex_expanded = _np.append(s_vertex, [1]).reshape(3, 1)
        s_vertex_pred = best_t.params.dot(s_vertex_expanded)[:2].reshape(-1)
        error = _np.linalg.norm(s_vertex_pred - t_vertex)

        # if s_i not in dict, or if its error is smaller than previous error
        if s_i not in inl_dict or (error < inl_dict[s_i][1]):
            inl_dict[s_i] = (t_i, error)
    inl_arr_unique = _np.array(
        [[s_i, t_i] for s_i, (t_i, e) in inl_dict.items()]
    )
    s, d = inl_arr_unique.T

    return best_t, (source_controlp[s], target_controlp[d])


def apply_transform(
    transform, source, target, fill_value=None, propagate_mask=False
):
    """Applies the transformation ``transform`` to ``source``.
    The output image will have the same shape as ``target``.
    Args:
        transform: A scikit-image ``SimilarityTransform`` object.
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.
    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.
    """
    from skimage.transform import warp

    if hasattr(source, "data") and isinstance(source.data, _np.ndarray):
        source_data = source.data
    else:
        source_data = source
    if hasattr(target, "data") and isinstance(target.data, _np.ndarray):
        target_data = target.data
    else:
        target_data = target

    aligned_image = warp(
        source_data,
        inverse_map=transform.inverse,
        output_shape=target_data.shape,
        order=3,
        mode="constant",
        cval=_np.median(source_data),
        clip=False,
        preserve_range=True,
    )
    footprint = warp(
        _np.zeros(source_data.shape, dtype="float32"),
        inverse_map=transform.inverse,
        output_shape=target_data.shape,
        cval=1.0,
    )
    footprint = footprint > 0.4

    if hasattr(source, "mask") and propagate_mask:
        source_mask = _np.array(source.mask)
        if source_mask.shape == source_data.shape:
            source_mask_rot = warp(
                source_mask.astype("float32"),
                inverse_map=transform.inverse,
                output_shape=target_data.shape,
                cval=1.0,
            )
            source_mask_rot = source_mask_rot > 0.4
            footprint = footprint | source_mask_rot
    if fill_value is not None:
        aligned_image[footprint] = fill_value

    return aligned_image, footprint


def register(source, target, fill_value=None, propagate_mask=False):
    """Transform ``source`` to coincide pixel to pixel with ``target``.
    Args:
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.
    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.
    """
    t, __ = find_transform(source=source, target=target)
    aligned_image, footprint = apply_transform(
        t, source, target, fill_value, propagate_mask
    )
    return aligned_image, footprint


def _find_sources(img):
    "Return sources (x, y) sorted by brightness."

    import sep

    if isinstance(img, _np.ma.MaskedArray):
        image = img.filled(fill_value=_np.median(img)).astype("float32")
    else:
        image = img.astype("float32")
    bkg = sep.Background(image)
    thresh = 3.0 * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh)
    sources.sort(order="flux")
    return _np.array([[asrc["x"], asrc["y"]] for asrc in sources[::-1]])

class MaxIterError(RuntimeError):
    pass

def _ransac(data, model, min_data_points, max_iter, thresh, min_matches):
    """fit model parameters to data using the RANSAC algorithm
This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
Given:
    data: a set of data points
    model: a model that can be fitted to data points
    min_data_points: the minimum number of data values required to fit the
        model
    max_iter: the maximum number of iterations allowed in the algorithm
    thresh: a threshold value to determine when a data point fits a model
    min_matches: the min number of matches required to assert that a model
        fits well to data
Return:
    bestfit: model parameters which best fit the data (or nil if no good model
              is found)
"""
    iterations = 0
    bestfit = None
    best_inlier_idxs = None
    n_data = data.shape[0]
    n = min_data_points
    all_idxs = _np.arange(n_data)

    while iterations < max_iter:
        # Partition indices into two random subsets
        _np.random.shuffle(all_idxs)
        maybe_idxs, test_idxs = all_idxs[:n], all_idxs[n:]
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) >= min_matches:
            betterdata = _np.concatenate((maybeinliers, alsoinliers))
            bestfit = model.fit(betterdata)
            best_inlier_idxs = _np.concatenate((maybe_idxs, also_idxs))
            break
        iterations += 1
    if bestfit is None:
        raise MaxIterError(
            "Max iterations exceeded while trying to find "
            "acceptable transformation."
        )

    return bestfit, best_inlier_idxs    
