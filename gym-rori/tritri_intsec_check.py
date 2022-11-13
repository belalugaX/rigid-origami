# a python implementation of Moeller's triangle-triangle intersection algorithm


import numpy as np



def edge_edge_test(v0,u0,u1,Ax,Ay):
    """
    this edge to edge test is based on Franlin Antonio's gem:
    "Faster Line Segment Intersection", in Graphics Gems III,
    pp. 199-202 -- Moeller's comment
    """
    Bx = u0[0]-u1[0]
    By = u0[1]-u1[1]
    Cx = v0[0]-u0[0]
    Cy = v0[1]-u0[1]
    f = Ay*Bx-Ax*By
    d = By*Cx-Bx*Cy
    if (f>0 and d>=0) or (f<0 and d<=0 and d>=f):
        e = Ax*Cy-Ay*Cx
        if f>0:
            if e>=0 and e<=f:
                return True
        else:
            if e<=0 and e>=f:
                return True
    return False

def edge_against_tri_edges(v0,v1,u0,u1,u2):
    Ax = v1[0]-v0[0]
    Ay = v1[1]-v0[1]
    # === test edge U0,U1 against V0,V1
    ret1 = edge_edge_test(v0,u0,u1,Ax,Ay)
    # === test edge U1,U2 against V0,V1
    ret2 = edge_edge_test(v0,u1,u2,Ax,Ay)
    # === test edge U2,U1 against V0,V1
    ret3 = edge_edge_test(v0,u2,u0,Ax,Ay)
    if (ret1 and ret2 and ret3):
        return True
    else:
        return False


def point_in_tri(v0,u0,u1,u2,i0,i1):
    # is T1 completely inside T2?
    # check if v0 is inside tri(u0,u1,u2)
    a = u1[i0]-u0[i1]
    b = -(u1[i0]-u0[i0])
    c = -a*u0[i0]-b*u0[i1]
    d0 = a*v0[i0]+b*v0[i1]+c

    a = u2[i1]-u1[i1]
    b = -(u2[i0]-u1[i0])
    c = -a*u1[i0]-b*u1[i1]
    d1 = a*v0[i0]+b*v0[i1]+c

    a = u0[i1]-u2[i1]
    b = -(u0[i0]-u2[i0])
    c = -a*u2[i0]-b*u2[i1]
    d2 = a*v0[i0]+b*v0[i1]+c

    if d0*d1>0 and d0*d2>0:
        return True
    else:
        return False


def coplanar_tri_tri(N,v0,v1,v2,u0,u1,u2):
    """
    first project onto an axis-aligned plane, that maximizes the area
    of the triangles, compute indices: i0,i1.
    """
    A = abs(N)
    print(N,A)
    if A[0]>A[1]:
        if A[0]>A[2]:
            # A[0] is greatest
            i0=1
            i1=1
        else:
            # A[2] is greatest
            i0 = 0
            i1 = 1
    else:
        if A[2]<=A[1]:
            # A[2] is greatest
            i0 = 0
            i1 = 1
        else:
            # A[1] is greatest
            i0=0
            i1=2

    # === test all edges of triangle 1 against the edges of triangle 2
    ret1 = edge_against_tri_edges(v0,v1,u0,u1,u2)
    ret2 = edge_against_tri_edges(v1,v2,u0,u1,u2)
    ret3 = edge_against_tri_edges(v2,v0,u0,u1,u2)
    if ret1 and ret2 and ret3:
        # === finally, test if tri1 is totally contained in tri2 or vice versa
        ret = point_in_tri(v0,u0,u1,u2,i0,i1)
        if ret: return True
        ret = point_in_tri(u0,v0,v1,v2,i0,i1)
        if ret: return True
    else: return False


def newcompute_intervals(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2):
    if D0D1>0:
        # --- here we know that D0D2<=0.0
        # --- that is D0, D1 are on the same side, D2 on the other or on the plane
        A=VV2
        B=(VV0-VV2)*D2
        C=(VV1-VV2)*D2
        X0=D2-D0
        X1=D2-D1
        return False,A,B,C,X0,X1

    elif D0D2>0:
        # --- here we know that d0d1<=0.0
        A=VV1
        B=(VV0-VV1)*D1
        C=(VV2-VV1)*D1
        X0=D1-D0
        X1=D1-D2
        return False,A,B,C,X0,X1

    elif D1*D2>0 or not D0==0:
        # --- here we know that d0d1<=0.0 or that D0!=0.0
        A=VV0
        B=(VV1-VV0)*D0
        C=(VV2-VV0)*D0
        X0=D0-D1
        X1=D0-D2
        return False,A,B,C,X0,X1

    elif not D1 == 0:
        A=VV1
        B=(VV0-VV1)*D1
        C=(VV2-VV1)*D1
        X0=D1-D0
        X1=D1-D2
        return False,A,B,C,X0,X1

    elif not D2 == 0:
        A=VV2
        B=(VV0-VV2)*D2
        C=(VV1-VV2)*D2
        X0=D2-D0
        X1=D2-D1
        return False,A,B,C,X0,X1

    else:
        # --- triangles are coplanar
        return True,0,0,0,0,0



def NoDivTriTriIsect(V0,V1,V2,U0,U1,U2):

    USE_EPSILON_TEST = True
    EPSILON = 0.000001

    # === 1. step: compute plane equation of triangle(V0,V1,V2)
    E1 = V1-V0
    E2 = V2-V0
    N1 = np.cross(E1,E2)
    d1 = -N1.dot(V0.T)

    # --- plane equation 1: N1.X+d1=0

    # --- put U0,U1,U2 into plane equation 1 to compute signed distances to the plane
    du0 = N1.dot(U0.T)+d1
    du1 = N1.dot(U1.T)+d1
    du2 = N1.dot(U2.T)+d1

    # --- coplanarity robustness check

    if USE_EPSILON_TEST:
        if(abs(du0)<EPSILON):
            du0=np.sign(du0)*0.0
        if(abs(du1)<EPSILON):
            du1=np.sign(du1)*0.0
        if(abs(du2)<EPSILON):
            du2=np.sign(du2)*0.0

    du0du1 = du0*du1
    du0du2 = du0*du2

    # === 2. step: Reject as trivial if all points of triangle 1 are on same side.
    if (du0du1>0 and du0du2>0) or (du0 == du1 == 0) or (du0 == du2 == 0) or (du1 == du2 ==0):
        # --- same sign on all of them + not equal 0 ?
        # no intersection occurs
        return False

    # === 3. step: compute plane of triangle (U0,U1,U2)
    E1 = U1-U0
    E2 = U2-U0
    N2 = np.cross(E1,E2)
    d2 = -N2.dot(U0.T)
    # --- plane equation 2: N2.X+d2=0

    # --- put V0,V1,V2 into plane equation 2
    dv0 = N2.dot(V0)+d2
    dv1 = N2.dot(V1)+d2
    dv2 = N2.dot(V2)+d2

    if USE_EPSILON_TEST:
        if(abs(dv0)<EPSILON):
            dv0=np.sign(dv0)*0.0
        if(abs(dv1)<EPSILON):
            dv1=np.sign(dv1)*0.0
        if(abs(dv2)<EPSILON):
            dv2=np.sign(dv2)*0.0

    dv0dv1 = dv0*dv1
    dv0dv2 = dv0*dv2

    # === 4. step: Reject as trivial if all points of triangle 2 are on same side.
    if (dv0dv1>0 and dv0dv2>0) or (dv0 == dv1 == 0) or (dv0 == dv2 == 0) or (dv1 == dv2 == 0):
        # --- same sign on all of them + not equal 0 ?
        # no intersection occurs
        return False


    # === 5. step: Compute intersection line and project onto largest axis.
    D = np.cross(N1,N2)

    # --- compute and index to the largest component of D
    max = abs(D[0])
    index = 0
    bb = abs(D[1])
    cc = abs(D[2])
    if bb>max:
        max = bb
        index = 1
    if cc>max:
        max = cc
        index = 2

    # --- this is the simplified projection onto L
    vp0 = V0[index]
    vp1 = V1[index]
    vp2 = V2[index]

    up0 = U0[index]
    up1 = U1[index]
    up2 = U2[index]

    # === 6. step: Compute the intervals for each triangle.
    # --- compute interval for triangle 1
    ret,a,b,c,x0,x1 = newcompute_intervals(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2)
    if ret:
        return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2)

    # --- compute interval for triangle 2
    ret,d,e,f,y0,y1 = newcompute_intervals(up0,up1,up2,du0,du1,du2,du0du1,du0du2)
    if ret:
        return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2)

    xx=x0*x1
    yy=y0*y1
    xxyy=xx*yy

    tmp=a*xxyy
    isect1 = np.ones(2)
    isect1[0]=tmp+b*x1*yy
    isect1[1]=tmp+c*x0*yy

    tmp=d*xxyy
    isect2 = np.ones(2)
    isect2[0]=tmp+e*xx*y1
    isect2[1]=tmp+f*xx*y0

    isect1 = np.sort(isect1)
    isect2 = np.sort(isect2)
    if (isect1[1]<=isect2[0]+EPSILON) or (isect2[1]<=isect1[0]+EPSILON):
        return False
    else:
        return True



def main():
    V0 = np.array([0,0,0])
    V1 = np.array([0,2.0,0])
    V2 = np.array([-2.0,0,0])

    U0 = np.array([0,0,0])
    U1 = np.array([0,2.0,0])
    U2 = np.array([0,1,1])
#    U0 = np.array([0,1,1])
#    U1 = np.array([-2,1,-1])
#    U2 = np.array([2,1,-1])
    print(NoDivTriTriIsect(V0,V1,V2,U0,U1,U2))
