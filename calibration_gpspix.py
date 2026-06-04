'''
Author: david
Date: 2026-05-08 14:46:09
LastEditTime: 2026-05-23 16:44:13
Description: 
'''
import numpy as np
import cv2
            
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    uB = np.mean(B, axis=0)
    uA = np.mean(A, axis=0)
    
    AA = A - uA
    BB = B - uB
    H = np.dot(AA.T , BB)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print ("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = np.dot(-R, uA.T) + uB.T

    return R, t

def estimate_homography(src_points, dst_points): # pix, gps
    assert len(src_points) == len(dst_points)

    # Create the matrix A
    num_points = len(src_points)
    A = np.zeros((2 * num_points, 9))
    for i in range(num_points):
        x, y = src_points[i]
        u, v = dst_points[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]

    # Solve homogeneous system of equations using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize H to have H[2, 2] = 1
    H /= H[2, 2]

    return H

def points_transfer_homography(matrix, points_to_trans, ref_points):
    assert len(points_to_trans) == len(ref_points)
    aa = np.dot(matrix, np.hstack((points_to_trans, np.ones((len(points_to_trans), 1)))).T) # pix2gps
    trans_res = (aa/aa[-1]).T[:,:2]
    err = trans_res-ref_points
    # import ipdb;ipdb.set_trace()
    err = np.sum(np.multiply(err,err))
    rmse = np.sqrt(err/len(points_to_trans))
    return  trans_res, rmse

if __name__=='__main__':


    gps_points = np.array([
        [118.829043, 31.962234],
        [118.829078, 31.962238],
        [118.829078, 31.96223],
        [118.829116, 31.962237],
        [118.829116, 31.962233],
        [118.829043, 31.962206],
        [118.829078, 31.962212],
        [118.829078, 31.962203],
        [118.829116, 31.96221],
        [118.829116, 31.962204],
        [118.829451, 31.96224],
        [118.829486, 31.962245],
        [118.829487, 31.962237],
        [118.829522, 31.962243],
        [118.829523, 31.962238],
        [118.829451, 31.962214],
    ])

    pix_points = np.array([
            [593, 1031],
            [586, 999],
            [647, 999],
            [628, 968],
            [650, 970],
            [931, 1028],
            [903, 994],
            [965, 995],
            [926, 966],
            [947, 966],
            [765, 804],
            [755, 791],
            [791, 792],
            [775, 780],
            [787, 781],
            [956, 803],
    ])

    gps_points_test = np.array([
            [118.829488, 31.962219],
            [118.829488, 31.96221],
            [118.829525, 31.962217],
            [118.829525, 31.962212],
    ])

    pix_points_test = np.array([
            [938, 789],
            [977, 790],
            [953, 778],
            [966, 778],
    ])


    # 谷歌地图测试点（自测）
    gps_points = [
    [118.829028, 31.962166],  
    [118.829026, 31.962193],
    [118.829026, 31.962220],
    [118.829026, 31.962247],

    [118.829042, 31.962208], 
    [118.829042, 31.962236], 

    [118.829114, 31.962208], 
    [118.829114, 31.962236], 

    [118.829021, 31.962120],
    [118.829023, 31.962089], 

    [118.829090, 31.962122],
    [118.829088, 31.962090],

    [118.829210, 31.962169], 
    [118.829351, 31.962172], 

    [118.828919, 31.962150], 

    [118.828958, 31.962166],
    [118.828954, 31.962192],

    [118.828955, 31.962225], 
    [118.828954, 31.962253],

    [118.828954, 31.962263],
    ];
    pix_points = [
        [1449, 1420],
        [1109, 1403],
        [759, 1404], 
        [421, 1405], 

        [935, 1386],
        [593, 1390], 

        [940, 1304],
        [636, 1309], 

        [2031, 1417],
        [2415, 1417],

        [1918, 1330],
        [2272, 1332],

        [1336, 1216],
        [1283, 1125],

        [1786, 1578],

        [1512, 1515],
        [1122, 1515],

        [645, 1507],
        [234, 1511],

        [98, 1512],
    ];
    # ret_H = rigid_transform_3D(A, B)
    # print(ret_H)

    # Estimate the homography matrix
    H = estimate_homography(pix_points, gps_points)
    H_inv = estimate_homography(gps_points, pix_points)


    print("Homography Matrix (H):")
    print(H)
    print(H_inv)
    print("-----------------------")


    trans_res, rmse = points_transfer_homography(H, pix_points_test, gps_points_test)
    print("pix2gps: %s"%trans_res)
    print("pix2gps err: %s" %rmse)
    print("-----------------------")

    trans_res, rmse = points_transfer_homography(np.linalg.inv(H), gps_points_test, pix_points_test)
    print("gps2pix: %s"%trans_res)
    print("gps2pix err: %s" %rmse)
    print("-----------------------")


    test_m = np.array([
    [0.01766573607, -0.2981458388, 118.8273383],
    [0.004751781665, -0.08019458623, 31.96204593], 
    [0.0001486648596, -0.002509045494, 1]
    ])
    trans_res, rmse = points_transfer_homography(test_m, pix_points_test, gps_points_test)
    print("[hurys] pix2gps: %s"%trans_res)
    print("[hurys] pix2gps err: %s" %rmse)
    print("-----------------------")

    trans_res, rmse = points_transfer_homography(np.linalg.inv(test_m), gps_points_test, pix_points_test)
    print("[hurys] gps2pix: %s"%trans_res)
    print("[hurys] gps2pix err: %s" %rmse)
    print("-----------------------")