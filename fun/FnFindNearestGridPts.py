def FnFindNearestGridPts(X, Y, Z, xlg, ylg, zlg):
# function to find the nearest grid positions for given points in space
# inputs:
# 	X, Y, Z  - grid domain size in x,y and z directions
# 	xlg, ylg, zlg - given points in space
# ouputs:
# 	Xd, Yd, Zd - nearest grid points
# 	pos - complete vector of grid points
# 	ZZ - index of nearest grid points
# 	k - x,y,z nearest grid positions
    import numpy as np
    from scipy.spatial import KDTree
    Yd, Zd, Xd = np.meshgrid(Y, Z, X, sparse=False, indexing='xy')
    # flatten the distance matrix as input to delauney
    pos = np.array([Xd.flatten(), Yd.flatten(), Zd.flatten()]).T
    # calculate euclidean distance matrix
    # DG = Delaunay(pos).simplices 

    ## nearest point search in the grid for Lidar measured points
    Lp = np.matrix([xlg.flatten(), ylg.flatten(), zlg.flatten()]).T
    # xy=[[x1,y1]...[xm,ym]] -> pos
    # XY=[[X1,Y1]...[Xm,Ym]] -> DG
    tree = KDTree(pos)
    dd, ii = tree.query(Lp, k=1, p=1, eps=0) # distance matrix, index
    ZZ = [] # index of the nearest point
    for i in range(len(dd)):
    	min_dd = np.min(dd[i])
    	min_dd_idx = np.where(dd[i]==min_dd)[0]
    	if len(min_dd_idx) > 1:
    		sorted_ii = np.sort(ii[i][min_dd_idx])
    		ZZ.append(sorted_ii[len(min_dd_idx)-1])
    		print('sorted list used, check the index')
    	else:
    #		ZZ.append(ii[i][0])
    		ZZ.append(ii[i]+1)
    k = pos[ZZ]
    return Xd, Yd, Zd, pos, ZZ, k
