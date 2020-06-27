import mascado.utility.affine as affine
import numpy as np
import pandas as pd
import os

def load_grid_data_D(ffolder, encoding='latin1'):
    """
    Parameters
    ----------
    ffolder : string
        Location of txt files with measured and reference positions..
    encoding : string
        File encoding, defaults to ``latin1``.
    Returns
    -------
    cats: ndarray
        pandas.DataFrame
        DataFrame with 4 columns:
        x-measured, y-measured
        x-reference , y-reference
    """
    
    entries = os.listdir(ffolder)
    cats = []
    
    for ent in entries:
        print(ent)
        data = np.genfromtxt(
            ffolder+'/'+ent, encoding=encoding,
            skip_header=1)
        df = pd.DataFrame(data, columns=[
            'x-measured', 'y-measured',
            'x-reference', 'y-reference'])
        cats.append(df)
    return cats

def distortions_of_image(cats, platescale=None, scale=1):
    r"""Get distortions and normalized positions of the image.
    By default, an affine transform is used for the transformation
    from reference coordinates to measured coordinates. The affine transform is the
    least-squares solution for each catalog.
    If ``platescale`` is passed, no affine transform, but
    a fixed scale is used.
    Normalized positions are calculated by shifting and scaling the
    positions into the domain :math:`[-1, 1]`.
    Parameters
    ----------
    cats : list of pandas.DataFrame
        Catalogs as described by doc of ``load_grid_data_D()``.
    platescale : float or None
        Optional fixed plate scale in pixel per mm.
    scale : float
        Additional, dimensionless scale apply applied to every position.
    Returns
    -------
    atrafo : (3, 3)-shaped array
        Affine transformation applied to translate from focal plane
        (measured coordinates) to mask.
    posscale : float
        Scale for normalized positions.
    positions : (N, 2)-shaped array
        Dimensionless, normalized positions.
    distortions : list of (N, 2)-shaped arrays
    posshift : float
        Shift for normalized positions.
    posmin : (1,2)-shaped array
        Minimum x and y measured position.
    posmax : (1,2)-shaped array
        Maximum x and y measured position.
    """

    #Get reference catalog
    refpos = [cats[i].loc[cats[i].index, ['x-reference', 'y-reference']].values for i in range(0,len(cats))]
    
    #Get measured catalog
    measpos = [cats[i].loc[cats[i].index, ['x-measured', 'y-measured']].values for i in range(0,len(cats))]
    
    #Calculate affin transformation
    if platescale is not None:
        atrafo = np.array([
            [platescale * scale, 0,                  0],
            [0,                  platescale * scale, 0],
            [0,                  0,                  1]])
    else:
        
        atrafo = np.array([affine.affine_lstsq(refpos[i], measpos[i]) for i in range(0,np.shape(refpos)[0])])
        
    trafpos = np.array([affine.affine_trafo(refpos[i], atrafo[i]) for i in range(0,np.shape(refpos)[0])])
    distortions = trafpos - measpos
    
    #Compute minimum/maximum x and y position
    posminx = np.min([pos[0] for mpos in measpos for pos in mpos])
    posminy = np.min([pos[1] for mpos in measpos for pos in mpos])
    posmaxx = np.max([pos[0] for mpos in measpos for pos in mpos])
    posmaxy = np.max([pos[1] for mpos in measpos for pos in mpos])
    
    pminx = [np.min(mpos[:,0]) for mpos in measpos]
    pminy = [np.min(mpos[:,1]) for mpos in measpos]
    pmin = np.array([pminx,pminy])
    
    pmaxx = [np.max(mpos[:,0]) for mpos in measpos]
    pmaxy = [np.max(mpos[:,1]) for mpos in measpos]
    pmax = np.array([pmaxx,pmaxy])
    
    posmin = np.array([posminx,posminy])
    posmax = np.array([posmaxx,posmaxy])
    
    #Normalize positions
    posscale = (posmax - posmin) / 2 
    posshift = (posmax + posmin) / 2
    positions = [(mpos - posshift) / posscale for mpos in measpos]    
     
    return atrafo, posscale, positions, distortions, posshift, pmin, pmax
