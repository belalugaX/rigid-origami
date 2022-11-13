# ------------------------------------------------------------------------------
#
#   PTU kinematics –– implementation of the PTU kinematic model
#   as discribed in paper:

#   Luca Zimmermann et al.:
#   "Conditions for Rigid and Flat Foldability of Degree-n Vertices in Origami"
#
#   author: Jeremia Geiger, Master Thesis: RL for Rigid Origami
#   under supervision of Oliver Richter and Karolis Martinkus
#   ETH Zurich, DISCO lab, December 2020
#
# ------------------------------------------------------------------------------




# ------------------------------- dependencies ---------------------------------

import numpy as np
from numpy import *
import vg

EPS = 1e-5
# ------------------------------------------------------------------------------
#
#   Rotation Matrix and forward and backward transformation definitions
#   for single vertex kinematics computation
#
# ------------------------------------------------------------------------------

# === rotation matrix about x-axis
def Rx(theta):
  return np.asarray([[ 1, 0           , 0     ],
                   [ 0, cos(theta),-sin(theta)],
                   [ 0, sin(theta), cos(theta)]])

# === rotation matrix about z-axis
def Rz(theta):
  return np.asarray([[ cos(theta), -sin(theta), 0 ],
                   [ sin(theta), cos(theta) , 0 ],
                   [ 0         , 0        , 1 ]])

# === forward rotatin, positive angles
def transform_fold_forw(angle_z,angle_x):
    return np.matmul(Rz(angle_z),Rx(angle_x))

# === reverse rotations, feed in negative angle values only!
def transform_fold_rev(angle_x,angle_z):
    return np.matmul(Rx(angle_x),Rz(angle_z))



def unit_vector(vector):
    # --------------------------------------------------------------------------
    #
    #   takes:      vector: an arbitrary 3D or 2D vector
    #
    #   returns:    the unit vector in 3D
    #
    # --------------------------------------------------------------------------

    np.seterr(invalid='ignore')  # ignore invalid cases

    if vector.shape[-1] == 2:
        vec_3D = np.insert(vector, 2, 0, axis=-1)  # 2D to 3D mapping
    else:
        vec_3D = vector

    nrm = np.linalg.norm(vec_3D, axis=-1, keepdims=True)
    nrm = np.where(nrm == 0, EPS, nrm)

    return vec_3D / nrm


def planar_angle(v1, v2):
    # --------------------------------------------------------------------------
    #
    #   takes:      v1,v2: two arbitraty 3D or 2D vectors
    #
    #   returns:    the signed 3D angle between two 3D vectors
    #
    # --------------------------------------------------------------------------

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    with np.errstate(divide='ignore'):
        angle = vg.signed_angle(v1_u,v2_u,look=vg.basis.z)/180*np.pi
        if isinstance(angle,np.ndarray):
            angle[np.isnan(angle)] = 0

    return angle


def compute_folded_unit(alpha_arr,rho):
    # === the first edge marks the x-axis of the SV-coordinate frame
    p_0 = np.array([1,0,0])
    new_sector = alpha_arr[-1]
    alpha = list(alpha_arr[:-1])
#    rho = list([rho.tolist()])
    res = np.identity(3)
    all_angles = zip(alpha,rho)#_arr[i,:])
    for a,r in all_angles:
        res = res.dot(transform_fold_forw(a,r))
    res = res.dot(Rz(new_sector))
    # === final 3D-point, marks the upper end of first unit
    p_m = np.matmul(res,p_0)
    # === return folded unit angle
    return np.arccos(np.clip(p_0.dot(p_m), -1., 1.))



def get_unit_angle(alpha_vec,rho_vec):
    # --- actual implementation of the kinematic model from PTU paper
    # --------------------------------------------------------------------------
    #
    #   takes:      alpha_vec: list of sector angles alpha
    #               rho_vec: list of dihedral angles rho
    #
    #
    #   returns:    U_j: unit angle U_j
    #               beta_j: start angle beta_j of unit
    #               delta_j: end angle delta_j of unit
    #
    # --------------------------------------------------------------------------

    if np.any(np.isnan(alpha_vec)):
        print('alpha nan', alpha_vec)
        raise NotImplementedError

    if np.any(np.isnan(rho_vec)):
        print('rho nan', rho_vec)
        raise NotImplementedError
    #starting point
    p_j0 = np.array([1,0,0])
    # number of sectors in unit
    m = len(alpha_vec)
    # rotation matrix placeholder
    Mffw = np.identity(3)
    # last unit-sector rotation
    last_rot_z = Rz(alpha_vec[-1])

    # === TODO: vectorize
    if m>1:
        # === case for non-rigid units
        # === forward kinematics for start angle beta
        for i in range(m-1):
            forw = transform_fold_forw(alpha_vec[i],rho_vec[i])
            Mffw = Mffw.dot(forw)
        Mffw = Mffw.dot(last_rot_z)
        p_jm = Mffw.dot(p_j0)
        if abs(p_j0.dot(p_jm)) > 1.:
            print('pjm', p_jm)
            print('pj0', p_j0)
        temp = clip(p_j0.dot(p_jm), -1., 1.)
        U_j = arccos(temp)
        if np.isnan(U_j):
            print('uj nan')
            raise NotImplementedError
        if abs(U_j)<EPS:
            #print("Unit Angle == 0",U_j)
            U_j = EPS
        p_j1 = np.matmul(Rz(alpha_vec[0]),p_j0)
        #start angle beta
        beta_j = beta_delta(p_j1,p_jm,U_j,alpha_vec[0])

        # === reverse kinematics for end angle delta
        Mffw = eye(3)
        p_jm_prime = p_j0.copy()
        Mffw = np.matmul(Rz(-alpha_vec[-1]),Mffw)
        alpha_vec_rev = alpha_vec.copy()
        alpha_vec_rev = alpha_vec_rev[:-1]
        rho_vec_rev = rho_vec.copy()
        alpha_vec_rev.reverse()
        rho_vec_rev.reverse()

        for i in range(m-1):
            Mffw = np.matmul(Mffw,transform_fold_rev(-rho_vec_rev[i],-alpha_vec_rev[i]))

        p_j0_prime = Mffw.dot(p_j0)
        p_jm1_prime = np.matmul(Rz(-alpha_vec[-1]),p_jm_prime)
        # end angle delta
        delta_j = beta_delta(p_jm1_prime,p_j0_prime,U_j,alpha_vec[-1])

    else:
        # --- case for rigid units only
        Mffw = last_rot_z
        p_jm = np.matmul(Mffw,p_j0)
        U_j = alpha_vec[0]
        gamma_sj = 0
        gamma_ej = 0

        # === start angle beta and end angle delta equal zero for 'rigid' units
        beta_j = 0
        delta_j = 0

    return [real(U_j),beta_j,delta_j]



def beta_delta(p_1,p_max,unit_angle,first_sector):
    # --------------------------------------------------------------------------
    #
    #   takes:      p_1: first next edge-point starting from a bdry unit point
    #               p_max: the last and max. unit point
    #               unit_angle: unit angle U_j(psi)
    #               first_sector: first sector angle alpha_i
    #
    #   returns:    res_angle: the start angle beta or end angle delta,
    #                   depending on the input points
    #
    # --------------------------------------------------------------------------
    gamma = arccos(clip(p_1.T.dot(p_max), -1., 1.))
    if np.isclose(
        p_max[2],0.,rtol=1e-05, atol=1e-08, equal_nan=False):
        p_max_z = 1e-05
    else:
        p_max_z = p_max[2]
    sgn = -np.sign(p_max_z)
    temp_enum = cos(gamma)-multiply(cos(first_sector),cos(unit_angle))
    temp_denom = multiply(sin(first_sector), sin(unit_angle))
    if abs(temp_denom) < EPS:
        temp_denom = sign(temp_denom) * EPS
    temp_denom = power(temp_denom,-1)
    temp = multiply(temp_enum,temp_denom)
    # if abs(temp)>1.0:
    #     print(temp)
    temp = clip(temp, -1., 1.)
    res_angle = sgn*arccos(temp)
    return res_angle



def get_theta(U):
    # --------------------------------------------------------------------------
    #
    #   takes:      U: a list of three unit angles U_j(psi), as a function of
    #                   the global driving angle psi
    #
    #   returns:    [theta_1,theta_2,theta_3]: a np array of three angles theta(psi)
    #
    # --------------------------------------------------------------------------

    def theta(u_i,u_j,u_k):
        ratio = multiply((cos(u_i)-multiply(cos(u_j),cos(u_k))),power(multiply(sin(u_j),sin(u_k)),-1))
        ratio = np.clip(ratio, -1., 1.)
        return arccos(ratio)
    theta_1 = theta(U[0],U[1],U[2])
    theta_2 = theta(U[1],U[0],U[2])
    theta_3 = theta(U[2],U[0],U[1])

#    print("theta",Matrix([theta_1,theta_2,theta_3]))

    return np.array([real(theta_1),real(theta_2),real(theta_3)])



def get_kin_data(
    incr,
    rbm,
    unit_sec_angles,
    dihedral_angles,
    init_rot=Rz(0),
    init_tilt=Rx(0)):
    # --------------------------------------------------------------------------
    #
    #   takes:      rbm: the rigid body mode [1,2]
    #               unit_sec_angles: list of sector angles alpha per unit
    #               dihedral_angles: list of dihedral angles rho per unit
    #               init_rot: the initial z-rotation matrix
    #
    #   returns:    phi: a list of three outgoing dihedral angles
    #               new_points_3D: list of three new 3D point coordinates,
    #                   as a function of (psi)
    #
    # --------------------------------------------------------------------------

    beta = []
    delta = []
    theta = []
    U = []
    multi_U = []
    multi_beta = []
    multi_delta = []
    multi_theta = []
    unit_points_3D = []
    p_j0 = np.array([[1],[0],[0]])
    dihedral_angles = np.flip(dihedral_angles,axis=0)
    triangle_inequality_holds = []
    for i in range(incr):
        U = []
        beta = []
        delta = []
        for j in range(3):
            alpha_vec = unit_sec_angles[j].copy()
            if j==0:
                rho_vec = dihedral_angles[i,:].copy().tolist()
            else:
                # === units 2 and 3 are rigid (no incoming creases/edges)
                rho_vec = []

            U_j,beta_j,delta_j = get_unit_angle(alpha_vec,rho_vec)
            U.append(U_j), beta.append(beta_j),delta.append(delta_j)
        multi_beta.append(beta)
        multi_delta.append(delta)
        multi_U.append(U)
        U_sorted = U.copy()
        U_sorted.sort()
        triangle_inequality = U_sorted[0]+U_sorted[1]>=U_sorted[-1]
        triangle_inequality_holds.append(triangle_inequality)
        if not triangle_inequality and len(triangle_inequality_holds)<incr:
            multi_U, multi_beta, multi_delta, triangle_inequality_holds = zero_padding(
                incr,multi_U,multi_beta,multi_delta,triangle_inequality_holds)
            break

    # reverse arrays to original order
    multi_U.reverse(), multi_beta.reverse(),multi_delta.reverse()
    triangle_inequality_holds.reverse()
    dihedral_angles = np.flip(dihedral_angles,axis=0)

    if np.any(np.isnan(multi_beta)):
        print('beta nan', multi_beta)
        raise NotImplementedError
    if np.any(np.isnan(multi_delta)):
        print('delta nan', multi_delta)
        raise NotImplementedError
    if np.any(np.isnan(multi_U)):
        print('U nan', multi_U)
        raise NotImplementedError

#    multi_U = np.where(np.array(multi_U)==0.0,0.0000001,multi_U) # numerical correction
    multi_phi = []
    for i in range(incr):
        theta = get_theta(multi_U[i])
        if np.any(np.isnan(theta)) and triangle_inequality_holds[i]:
            print('nan theta found', theta)
            print('even with triangle inequality fullfilled!!')
            print(multi_U[i])
        multi_phi.append(get_phi(rbm,multi_beta[i],multi_delta[i],theta))
    multi_phi = np.array(multi_phi).reshape((-1,3))
    multi_phi = np.where(multi_phi<-np.pi,2*np.pi+multi_phi,multi_phi)
    multi_phi = np.where(multi_phi>np.pi,-2*np.pi+multi_phi,multi_phi)
    multi_phi = np.roll(multi_phi,1,axis=1)
    dihedral_angles = np.hstack((dihedral_angles,multi_phi))
    new_points_3D, point_transforms = get_new_points(incr,unit_sec_angles,dihedral_angles.copy(),multi_phi,init_rot,
                                                     p_j0,triangle_inequality_holds)
    return multi_phi,new_points_3D,point_transforms,multi_U,dihedral_angles,triangle_inequality_holds

def zero_padding(incr, multi_U, multi_beta, multi_delta, triangle_inequality_holds):
    missing_elements = np.zeros(incr-len(triangle_inequality_holds)).astype(bool)
    triangle_inequality_holds = np.hstack(
        (triangle_inequality_holds,missing_elements)).tolist()
    assert len(triangle_inequality_holds)==incr, "length of triangle_inequality_holds not same incr"
    padding = np.ones((incr,3))
    padding[:np.shape(multi_U)[0],:] = np.array(multi_U)
    multi_U = padding.tolist()
    padding[:np.shape(multi_beta)[0],:] = np.array(multi_beta)
    multi_beta = padding.tolist()
    padding[:np.shape(multi_delta)[0],:] = np.array(multi_delta)
    multi_delta = padding.tolist()
    return multi_U, multi_beta, multi_delta, triangle_inequality_holds

def get_phi(rbm,beta,delta,theta):
    # --------------------------------------------------------------------------
    #
    #   takes:      rbm: the rigid body mode
    #               beta: list of starting angles beta
    #               delta: list of end angles delta
    #               theta: list of theta angles
    #
    #   returns:    phi: list of three outgoing edge dihedral angles phi
    #
    # --------------------------------------------------------------------------

    if rbm == 1:
        phi1 = beta[2] + pi - theta[0] + delta[1]
        phi2 = beta[0] + pi - theta[1] + delta[2]
        phi3 = beta[1] + pi - theta[2] + delta[0]
    else:
        phi1 = beta[2] + theta[0] - pi + delta[1]
        phi2 = beta[0] + theta[1] - pi + delta[2]
        phi3 = beta[1] + theta[2] - pi + delta[0]

    return np.array([real(phi1),real(phi2),real(phi3)])


def get_new_points(incr,sec_angles,dih_angles,phi,init_rot,p_j0,triangle_ineq_holds):
    # --------------------------------------------------------------------------
    #
    #   takes:      sec_angles: list of sector angles
    #               dih_angles: list of dihedral all_angles
    #               phi: calculated phi angles
    #               init_rot: the initial z-rotation matrix
    #               p_j0: the first unit point p_j0 (2D)
    #
    #   returns:    points_3D: a list of 3D coordinates of the three new
    #                   outgoing vertices[[x_1(psi),y_1(psi),z_1(psi)],...]
    #               point_transforms: list of init_transforms to pass on to
    #                   child vertices
    #
    # --------------------------------------------------------------------------
    flatten = lambda x: [elem for sublist in x for elem in sublist]
    start_pos = 1
    s = flatten(sec_angles)
    s = np.roll(np.array(s),-start_pos)    #start_pos should always be one
    d = np.roll(dih_angles,-start_pos)
    d = np.insert(d[:,:-1],0,0,axis=1)

    multi_points_3D = []
    multi_point_transforms = []
    reverse_orient = Rz(np.pi)

    for i in range(incr):
        rhos = d[i,:]
        i_rot = np.array(init_rot[i])
        point_transforms = []
        points_3D = i_rot.dot(np.array([[1],[0],[0]])).reshape((3,1))
        Mffw = Rz(0)
        for alpha,rho in zip(s,rhos):
            Mffw = Mffw.dot(Rx(rho).dot(Rz(alpha)))
            new_point_sv = Mffw.dot(p_j0)
            new_point_3D = np.matmul(i_rot,new_point_sv).reshape((3,1))
            if triangle_ineq_holds[i] and np.any(np.isnan(new_point_3D)):
                print(new_point_3D)
                print(Mffw)
                print(alpha)
                print(rho)
                print('---')
            points_3D = np.hstack((points_3D,new_point_3D))
            rot2pass = i_rot.dot(Mffw)
            point_transforms.append(rot2pass.dot(reverse_orient))
        multi_points_3D.append(points_3D.transpose())
        multi_point_transforms.append(point_transforms)

    return multi_points_3D, multi_point_transforms
