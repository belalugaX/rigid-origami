import numpy as np
from numpy import pi,cos,sin,array,matmul
from kinematic_model_num import planar_angle, Rx, Rz, get_kin_data
from find_cycles import find_cycles
from tritri_intsec_check import NoDivTriTriIsect
import matplotlib.pyplot as plt

def search_bnd_vtx_within_symmetry(env, start_vtx, upper_bound_bln: bool):
# ------------------------------------------------------------------------------
#
#   Search for the next boundary vertex within the symmetry bounds
#
#   takes:      vtx: vertex2extend obj
#               upper_bound_bln:    True for searching upper (CW) boundary vertex,
#                                   False for lower (CCW) boundary vertex
#
#   returns:    bnd_vtx: name of boundary vertex or vertex on symmetry axis
#               bool: whether the vertex was found within the symmetry
#
# ------------------------------------------------------------------------------
    idx_delta = -1 if upper_bound_bln else +1
    start_on_symmetry_axis = is_on_symmetry_axis(env, start_vtx.name)
    vertex_to_check = start_vtx.surrounding_vertices[-1] if upper_bound_bln else start_vtx.surrounding_vertices[0]
    vtx_obj = env.vertex_objs[vertex_to_check]
    idx = vtx_obj.surrounding_vertices.index(start_vtx.name)
    previous_vtx = start_vtx.name
    while True:
        if crossed_symmetry_axis(env, start_vtx, vtx_obj):
            return previous_vtx, True
        elif not vtx_obj.is_extended:
            return vertex_to_check, False
        else:
            if start_on_symmetry_axis and not is_on_symmetry_axis(env, vertex_to_check):
                start_vtx = vtx_obj
            num_sv = len(vtx_obj.surrounding_vertices)
            previous_vtx = vertex_to_check
            vertex_to_check = vtx_obj.surrounding_vertices[(idx+idx_delta)%num_sv]
            new_vtx_obj = env.vertex_objs[vertex_to_check]
            idx = new_vtx_obj.surrounding_vertices.index(vtx_obj.name)
            vtx_obj = new_vtx_obj

def crossed_symmetry_axis(env, start_vtx, vtx_obj):
    if not env.symmetry:
        return False
    xs,ys = env.points_2D[start_vtx.name]
    xn,yn = env.points_2D[vtx_obj.name]
    if env.symmetry_axis == 'x-axis':
        return ys * yn < 0
    elif env.symmetry_axis == 'y-axis':
        return xs * xn < 0
    elif env.symmetry_axis == 'point':
        tmp_bool = False
        if env.cross_axis_symmetry:
            tmp_bool = (xs - ys) * (xn - yn) < 0
        return xs * xn < 0 or ys * yn < 0 or tmp_bool
    else:
        raise NotImplementedError

def is_on_symmetry_axis(env, vtx):
    if not env.symmetry:
        return False
    x,y = env.points_2D[vtx]
    if env.symmetry_axis == 'x-axis':
        return y == 0.
    elif env.symmetry_axis == 'y-axis':
        return x == 0.
    elif env.symmetry_axis == 'point':
        tmp_bool = False
        if env.cross_axis_symmetry:
            tmp_bool = abs(x)==abs(y)
        return x==0. or y==0 or tmp_bool
    else:
        raise NotImplementedError

class PseudoVertex():
    def __init__(self, env, name, key, coords_2D, coords_3D, neighbor):
        self.name = name
        self.key = key
        self.coords = coords_2D
        self.coords_3D = coords_3D
        self.neighbor = neighbor
        self.is_pseudo = True
        self.inactive = False
        self.env = env
        env.vertex_pos_list.append(self.__map2board(coords_2D))

    def __map2board(self,new_points_2D):

        new_vertex_pos = []
        if len(new_points_2D)==2:
            new_vertex_pos = [int(self.env.half_board_length - new_points_2D[1]), int(self.env.half_board_length + new_points_2D[0])]
        else:
            for coord_x,coord_y in new_points_2D:
                new_vertex_pos.append([int(self.env.half_board_length - coord_y), int(self.env.half_board_length + coord_x)])

        return new_vertex_pos


class SingleVertex():
# ------------------------------------------------------------------------------
#
#   A Single Vertex (SV) is the basic primitive of a PTU based rigid origami
#   model and crease pattern. A PTU rigid origami crease pattern consists of
#   multiple interlinked SV's.
#   SV's are linked by edges and inherit dihedral angle values phi to child SV's.
#
#   properties:     name (int)
#                   coords [x,y]
#                   coords_3D [3,float]
#                   mother_vertex [int]
#                   driving_angle [float list]
#                   surrounding_vertices [int list]
#                   surrounding_c_len [float list]
#                   rbm [1,2]
#                   phi [3,float]
#                   polygons [int list]
#                   init_rot [3x3 array]
#                   extended (boolean)
#
#   public methods: extend(rbm,ext_vertex_coords)
#
# ------------------------------------------------------------------------------

    def __init__(   self,
                    env_obj,
                    name,
                    coords,
                    mother_vertex_name,
                    driving_angle,
                    c_len=1,
                    init_rot=Rz(0),
                    units=[],
                    mirrored=False,
                    new_vertex=True,
                    rbm=1):
        # ----- initialize single vertex properties -----
        self.env = env_obj
        self.name = name
        self.is_pseudo = False
        self.coords = np.asarray(coords)
        self.coords_3D = np.array([[coords[0],coords[1],0] for _ in range(len(self.env.psi_opt))])
        self.mother_vertex = mother_vertex_name
        self.init_rot = init_rot if isinstance(init_rot,list) else [init_rot for _ in range(len(self.env.psi_opt))]
        self.board_pos = self.__map2board(coords)
        self.surrounding_vertices = [mother_vertex_name]
        self.surr_c_len = [c_len]
        self.unit_angles = []
        self.sector_angles = []
        self.dihedral_angles = driving_angle
        self.units = units
        self.is_extended = False
        self.is_mirrored = mirrored
        self.rbm = rbm
        self.__update_board_state([self.board_pos])
        self.update_adjacency_state([[name,mother_vertex_name]])
        if new_vertex:
            self.env.vertex_pos_list.append(self.board_pos)
            self.env.points_2D.append(coords)
        self.polygons = []

# ------------------------------------------------------------------------------
# ----- public methods ---------------------------------------------------------
# ------------------------------------------------------------------------------

    def extend(self,rbm,ext_vertex_coords_ro):
    # --------------------------------------------------------------------------
    #
    #   takes:      rbm: rigid body mode (int)
    #               ext_vertex_coordinates: a list of 2D coordinates for the
    #                   three new extension vertices
    #
    #   ->          creates and initializes three new vertices from self
    #   ->          updates properties of surrounding vertices
    #   ->          updates env variables, including state
    #
    #   returns:    -
    #
    # --------------------------------------------------------------------------

        ext_vertex_names = []
        ext_board_coords = self.__map2board(ext_vertex_coords_ro)
        new_i = []
        upgrade_i = []
        merge_i = []
        c_len = []
        self.new_faces = []
        self.env.face_boundary_edges = [edge for edge in self.env.face_boundary_edges if self.name not in edge]
        vertex_list_orig = self.env.vertex_list.copy()
        vertex_objs_orig = self.env.vertex_objs.copy()

        for i in range(len(ext_vertex_coords_ro)):
#            ii,jj = ext_board_coords[i]
            # === test for merge candidates
            if  ext_vertex_coords_ro[i] in self.env.points_2D: #not self.env.state[ii,jj,0] == 0:
                vtx_obj = self.env.vertex_objs[self.env.points_2D.index(ext_vertex_coords_ro[i])]
                ext_vertex_names.append(vtx_obj.name)#self.env.vertex_pos_list.index(ext_board_coords[i])) #find existing vertex
                if vtx_obj.is_pseudo and not vtx_obj.inactive:
                    upgrade_i.append(i)
                else:
                    merge_i.append(i)
            else:
                new_vertex = len(self.env.vertex_list)
                self.env.vertex_list.append(new_vertex)
                ext_vertex_names.append(new_vertex)
                new_i.append(i)
            c_len.append(np.linalg.norm(ext_vertex_coords_ro[i]-self.coords))

        # --- search boundary vertices for outer faces (before updating surrounding vertices of self)
        bnd_vtx_zip = []
        for idx, vtx in enumerate(ext_vertex_names[:-1]):  # nothing to do on middle extended node (at idx=2)
            bnd_vtx, on_symmetry = search_bnd_vtx_within_symmetry(self.env, self, idx != 0)
            bnd_vtx_zip.append((bnd_vtx, on_symmetry, vtx))

        # --- update list of surrounding vertices
        self.surrounding_vertices.insert(0,ext_vertex_names[0])
        self.surrounding_vertices.extend(ext_vertex_names[1:])
        self.surr_c_len.insert(0,c_len[0])
        self.surr_c_len.extend(c_len[1:])

        # --- new vertex instances
        # self.env.len_new_vertices = len(new_i)  # TODO necessary?
        for i in new_i:
            self.env.vertex_objs.append(SingleVertex(self.env, ext_vertex_names[i], ext_vertex_coords_ro[i],
                                                     self.name, 0, c_len[i]))
        for i in upgrade_i:
            key = self.env.vertex_objs[ext_vertex_names[i]].key
            self.env.vertex_objs[ext_vertex_names[i]] = SingleVertex(self.env, ext_vertex_names[i],
                                                                     ext_vertex_coords_ro[i], self.name, 0,
                                                                     c_len[i], new_vertex=False)
            del self.env.pseudo_vertices[key]


        # --- determine single vertex PTU kinematics and derive output folding angles phi
        self.rbm = rbm
        units,phi,multi_new_points_3D,multi_point_transforms,U = \
            self.__single_vertex_kinematics(c_len)
        polygons = self.__get_polygon_list(self.surrounding_vertices)

        # --- update board state
        self.__update_board_state([self.board_pos],rbm)
        # --- reset selection
        self.env.state[:,:,-1] = 0

        # --- update unit property
        self.units = units

        # --- update existing adjacent vertices
        new_edges = [[self.name,name] for name in ext_vertex_names]
        self.update_adjacency_state(new_edges)

        for i in merge_i:
            if i==0:
                is_u_bnd = True
            else:
                is_u_bnd = False
            self.__update_merged_vertex(ext_vertex_names[i],multi_point_transforms[i],self.phi[:,i],c_len[i],new_edges,is_u_bnd)

        for i in new_i:
            self.__update_new_vertex(ext_vertex_names[i],multi_new_points_3D[i,:,:].T,multi_point_transforms[i],self.phi[:,i],new_edges)
        for i in upgrade_i:
            self.__update_new_vertex(ext_vertex_names[i],multi_new_points_3D[i,:,:].T,multi_point_transforms[i],self.phi[:,i],new_edges)

        # --- update edge_list
        for v in ext_vertex_names:
            self.env.edge_list.append([self.name,v])

        # --- add faces between extended vertices
        self.new_faces.append([self.name, ext_vertex_names[0], ext_vertex_names[2]])
        self.env.face_boundary_edges.append([ext_vertex_names[0], ext_vertex_names[2]])
        self.new_faces.append([self.name, ext_vertex_names[1], ext_vertex_names[2]])
        self.env.face_boundary_edges.append([ext_vertex_names[1], ext_vertex_names[2]])

        # --- add outer faces
        for bnd_vtx, on_symmetry, vtx in bnd_vtx_zip:
            if on_symmetry:
                bnd_vtx = self.__create_or_update_pseudo_vtx(bnd_vtx, vtx)
            if not bnd_vtx == vtx: # no face needs to be added on merges
                if vtx in [ext_vertex_names[i] for i in upgrade_i]: continue # upgraded vertices don't need this face
                self.new_faces.append([self.name, vtx, bnd_vtx])
                self.env.face_boundary_edges.append([vtx, bnd_vtx])

        self.polygons = polygons # TODO necessary?

        self.env.triangles.extend(self.new_faces)
        # === no foldability check per default if no numerical optimal
        # === fold angle psi_opt specified
        foldable = True
        msg = "no foldabilty check for non-symbolic evaluation"
        self.is_extended = True
        foldable, msg = self.check_foldability(ext_vertex_names)  # polygons)
        return foldable, msg


# ------------------------------------------------------------------------------
# ----- private methods ---------------------------------------------------------
# ------------------------------------------------------------------------------

    def __map2board(self,new_points_2D):

        new_vertex_pos = []
        if len(new_points_2D)==2:
            new_vertex_pos = [int(self.env.half_board_length - new_points_2D[1]), int(self.env.half_board_length + new_points_2D[0])]
        else:
            for coord_x,coord_y in new_points_2D:
                new_vertex_pos.append([int(self.env.half_board_length - coord_y), int(self.env.half_board_length + coord_x)])

        return new_vertex_pos


    def __update_board_state(self,new_vertex_pos,rbm=1):
        if rbm >= 0:
            board_rbm = 1
        else:
            board_rbm = -1
        for coords_x,coords_y in new_vertex_pos:
            self.env.state[coords_x,coords_y,0] = board_rbm
        return self.env.state[:,:,0]


    def update_adjacency_state(self,new_edges):
        for v_pos_centr,v_pos_out in new_edges:
            if self.name in (v_pos_centr,v_pos_out):
                if v_pos_out == self.name:
                    self.env.state[self.board_pos[0],self.board_pos[1],(v_pos_centr+1)] = -1
                else:
                    self.env.state[self.board_pos[0],self.board_pos[1],int(v_pos_out+1)] = 1


    def __update_merged_vertex(self,v,start_rot,inc_phi,c_len,new_edges,is_u_bnd):
        vtx = self.env.vertex_objs[v]
        vtx.update_adjacency_state(new_edges)

        inc_phi = inc_phi.reshape((-1,1))

        # u_bnd for upper bound appending, False otherwise
        if is_u_bnd:
            vtx.surrounding_vertices.append(self.name)
            vtx.dihedral_angles = np.hstack((vtx.dihedral_angles,inc_phi))
            vtx.surr_c_len.append(c_len)

        else:
            vtx.surrounding_vertices.insert(0,self.name)
            vtx.dihedral_angles = np.hstack((inc_phi,vtx.dihedral_angles))
            vtx.surr_c_len.insert(0,c_len)
            vtx.init_rot = start_rot

        # MIURA ORI
        # if len(vtx.surrounding_vertices) == 2:
        #     # if True: check whether edges colinear, than mark as is_extended
        #     # this will allow the modeling of patterns like the Miura Ori
        #     a,b = vtx.surrounding_vertices
        #     coords_a, coords_b = self.env.points_2D[a], self.env.points_2D[b]
        #     if coords_a[0] == coords_b[0] == vtx.coords[0] or \
        #         coords_a[1] == coords_b[1] == vtx.coords[1]:
        #         vtx.is_extended = True


    def __update_new_vertex(self, v: int, point_3D, init_rot, inc_phi, new_edges):
        vtx = self.env.vertex_objs[v]
        vtx.coords_3D = point_3D
        vtx.init_rot = init_rot
        vtx.dihedral_angles = inc_phi.reshape((-1,1))
        vtx.update_adjacency_state(new_edges)


    def __create_or_update_pseudo_vtx(self, bnd_vtx, vtx):
        x, y = self.env.points_2D[bnd_vtx]
        vtx_x, vtx_y = self.env.points_2D[vtx]
        sym_axis = self.env.symmetry_axis
        switchpoint = sum(self.env.points_2D[0])
        if self.env.symmetry_axis == 'point':
            switchpoint = 0
            if x == 0:
                sym_axis = 'y-axis'
            elif y == 0:
                sym_axis = 'x-axis'
            elif x == y:
                sym_axis = 'xy'
            else:
                raise NotImplementedError
        if sym_axis == 'x-axis':
            assert y == 0
            key = 'x+' if vtx_x > switchpoint else 'x-'
            coords_2D = vtx_x, 0.
            coords_3D = self.env.vertex_objs[vtx].coords_3D.copy()
            coords_3D[:, 1] = 0.
            return self.__pseudo_vtx_helper(key, bnd_vtx, vtx, coords_2D, coords_3D)
        elif sym_axis == 'y-axis':
            assert x == 0
            key = 'y+' if vtx_y > switchpoint else 'y-'
            coords_2D = 0., vtx_y
            coords_3D = self.env.vertex_objs[vtx].coords_3D.copy()
            coords_3D[:, 0] = 0.
            return self.__pseudo_vtx_helper(key, bnd_vtx, vtx, coords_2D, coords_3D)
        elif sym_axis == 'xy':
            assert x == y
            if vtx_x > switchpoint and vtx_y > switchpoint:
                key = 'xy++'
            elif vtx_x > switchpoint and vtx_y < switchpoint:
                key = 'xy+-'
            elif vtx_x < switchpoint and vtx_y > switchpoint:
                key = 'xy-+'
            else:
                key = 'xy--'
            unsigned_average = 0.5 * (abs(vtx_x) + abs(vtx_y))
            coords_2D = np.sign(vtx_x) * unsigned_average, np.sign(vtx_y) * unsigned_average
            coords_3D = self.env.vertex_objs[vtx].coords_3D.copy()
            unsigned_averages = 0.5 * (np.abs(coords_3D[:, 0]) + np.abs(coords_3D[:, 1]))
            coords_3D[:, 0] = np.sign(coords_3D[:, 0]) * unsigned_averages
            coords_3D[:, 1] = np.sign(coords_3D[:, 1]) * unsigned_averages
            return self.__pseudo_vtx_helper(key, bnd_vtx, vtx, coords_2D, coords_3D)
        raise NotImplementedError  # currently, cross symmetry not yet implemented


    def __pseudo_vtx_helper(self, key, bnd_vtx, vtx, coords_2D, coords_3D):
        coords_2D = [int(coords_2D[0]), int(coords_2D[1])]
        if key in self.env.pseudo_vertices:
            # update pseudo vertex
            pseudo_vtx, msg = self.env.pseudo_vertices[key]
            if msg == 'del':  # pseudo vertex marked for deletion if vtx on symmetry axis
                del self.env.pseudo_vertices[key]  # this ensures a new face gets a new pseudo vertex
                return vtx
            elif msg == 'new':
                # if we create a pseudo vertex, we also have to add the pseudo vertex face
                self.new_faces.append([pseudo_vtx, bnd_vtx, vtx])
                self.env.face_boundary_edges.append([pseudo_vtx, vtx])
                self.env.pseudo_vertices[key] = (pseudo_vtx, '')
                return bnd_vtx  # return bnd_vtx here, such that outerface gets filled to bound vtx

            # update
            self.env.points_2D[pseudo_vtx] = coords_2D
            self.env.vertex_objs[pseudo_vtx].coords = coords_2D
            self.env.vertex_objs[pseudo_vtx].coords_3D = coords_3D
            self.env.vertex_objs[pseudo_vtx].neighbor = vtx
            self.env.vertex_pos_list[pseudo_vtx] = self.__map2board(coords_2D)

            if is_on_symmetry_axis(self.env, vtx):
                # self.env.vertex_objs[pseudo_vtx] = self.env.vertex_objs[vtx]  # refer full vertex
                self.env.vertex_objs[pseudo_vtx].name = vtx
                self.env.vertex_objs[pseudo_vtx].inactive = True
                self.env.pseudo_vertices[key] = (pseudo_vtx, 'del')  # delete pseudo vertex after symmetry expansion
                return vtx
            return pseudo_vtx
        elif is_on_symmetry_axis(self.env, vtx):
            # no need to create a pseudo vertex
            return bnd_vtx  # return bnd_vtx here, such that outerface gets filled to bound vtx
        else:
            pseudo_vtx = len(self.env.vertex_list)
            self.env.vertex_list.append(pseudo_vtx)
            self.env.points_2D.append(coords_2D)
            self.env.vertex_objs.append(PseudoVertex(self.env, pseudo_vtx, key, coords_2D, coords_3D, vtx))
            self.env.vertex_pos_list.append(self.__map2board(coords_2D))
            # print('created pseudo node ' + key, pseudo_vtx, coords_2D, coords_3D)
            self.env.pseudo_vertices[key] = (pseudo_vtx, 'new')
            # if we create a pseudo vertex, we also have to add the pseudo vertex face
            self.new_faces.append([pseudo_vtx, bnd_vtx, vtx])
            self.env.face_boundary_edges.append([pseudo_vtx, vtx])
            return bnd_vtx  # return bnd_vtx here, such that outerface gets filled to bound vtx

    def __single_vertex_kinematics(self, c_len: list):
    # --------------------------------------------------------------------------
    #
    #   takes:      c_len: a list of adjacent edge lengths, starting from the
    #                   first unit vertex
    #
    #   returns:    self.units: unit angles of single vertex
    #               phi_vec: list of three outgoing edge driving angles
    #               [point_1,point_2,point_3]: new outgoing point 3D coordinates
    #
    # --------------------------------------------------------------------------

        # --- get sector angles
        sector_angles = []
        for i in range(len(self.surrounding_vertices)-1):
            sector_angles.append(planar_angle(\
            self.env.vertex_objs[self.surrounding_vertices[i]].coords - self.coords,\
            self.env.vertex_objs[self.surrounding_vertices[i+1]].coords - self.coords))
        sector_angles.append(planar_angle(\
        self.env.vertex_objs[self.surrounding_vertices[-1]].coords - self.coords,\
        self.env.vertex_objs[self.surrounding_vertices[0]].coords - self.coords))

        # --- assemble sector angles list per unit
        self.units = [sector_angles[0:-2],[sector_angles[-2]],[sector_angles[-1]]]

        # starting edge for forward kinematics
        #should always set equal to one -> first incoming edge on lower bound
        start_pos = 1

        num_valid_fold_angles = np.count_nonzero(self.env.triangle_inequality_holds)
        diff = self.env.triangle_inequality_holds.size - num_valid_fold_angles
        init_rot = np.array(self.init_rot.copy()[diff:])

        # --- get unknown dihedral angles phi
        phi,self.out_points_3D,point_transforms,U,self.dihedral_angles,triangle_inequality_holds = \
            get_kin_data(
                num_valid_fold_angles,
                self.rbm,
                self.units.copy(),
                self.dihedral_angles.copy()[self.env.triangle_inequality_holds],
                init_rot)

        # --- assemble list of 3D points dependent on enval driving angle psi
        multi_transforms2pass = [[],[],[]]
        for j in range(diff):
            multi_transforms2pass[0].append([None])
            multi_transforms2pass[1].append([None])
            multi_transforms2pass[2].append([None])
        multi_points2pass = np.ones((3,3,len(self.env.psi_opt)))
        for i in range(num_valid_fold_angles):
            j = i+diff
            multi_points2pass[0,:,j] = c_len[0]*(self.out_points_3D[i][-1-start_pos])+self.coords_3D[j,:]
            multi_points2pass[1,:,j] = c_len[1]*(self.out_points_3D[i][-3-start_pos])+self.coords_3D[j,:]
            multi_points2pass[2,:,j] = c_len[2]*(self.out_points_3D[i][-2-start_pos])+self.coords_3D[j,:]
            multi_transforms2pass[0].append(point_transforms[i][-1-start_pos])
            multi_transforms2pass[1].append(point_transforms[i][-3-start_pos])
            multi_transforms2pass[2].append(point_transforms[i][-2-start_pos])

        self.U = U
        # make sure triangle inequality holds also for prior extensions
        triangle_inequality_holds = np.hstack((np.zeros(diff),triangle_inequality_holds))
        self.env.triangle_inequality_holds = \
            np.logical_and(triangle_inequality_holds,self.env.triangle_inequality_holds)
        if diff>0:
            phi = np.vstack((np.zeros((diff,np.shape(phi)[1])),phi))
        new_phi = np.roll(phi,1,axis=1)
        self.phi = new_phi

        return self.units,new_phi,multi_points2pass,multi_transforms2pass,U


    def __get_polygon_list(self, verts: list):
    # --------------------------------------------------------------------------
    #
    #   takes:      verts: a list of surrounding vertices of a single vertex
    #
    #   returns:    poly_verts: a list of triangular face definitions
    #                   [vertex_#,vertex_#,vertex_#],
    #                   each adjacent to the central single vertex
    #
    # --------------------------------------------------------------------------

        poly_verts = np.zeros((len(verts),3),dtype=int)
        poly_verts[:,0] = verts
        poly_verts[:-1,1] = verts[1:]
        poly_verts[-1,1] = verts[0]
        poly_verts[:,2] = [self.name for _ in range(len(verts))]

        return poly_verts


    def check_foldability(self, new_vertices):  # sv_faces: list):
        # === triangle inequality check if non optimize selected
        if np.all(~np.array(self.env.triangle_inequality_holds)) or (not self.env.optimize_psi and \
            not np.all(self.env.triangle_inequality_holds)):
            return False, "Triangle Inequality does not hold: U_max > U_min + U_med"

        # === global intersection test: triangle-triangle-intersection
        if self.tritri_isec_check(new_vertices):  #sv_faces):
            return False, "intersecting faces"
        else:
            return True, "all tests passed"

    @staticmethod
    def point_in_triangle(P, triangle):
        A, B, C = triangle
        v0 = [C[0] - A[0], C[1] - A[1]]
        v1 = [B[0] - A[0], B[1] - A[1]]
        v2 = [P[0] - A[0], P[1] - A[1]]
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        u = cross(v2, v0)
        v = cross(v1, v2)
        d = cross(v1, v0)
        if d < 0:
            u, v, d = -u, -v, -d
        return u > 0 and v > 0 and (u + v) < d


    def tritri_isec_check(self, new_vertices):  #sv_faces=[]):
        # === triangle-triangle intersection test
        max_fold_angle_index = np.min(np.where(self.env.triangle_inequality_holds))
        all_3D_points = self.env.all_3D_points.swapaxes(0,1)
        points_2D = np.asarray(self.env.points_2D)
        for idx in range(len(all_3D_points)-1, max_fold_angle_index-1, -1):
            points = all_3D_points[idx]
            for face in self.env.triangles:
                for new_face in self.new_faces:
                    if self.tri_check(face, new_face, points[face], points[new_face]):
                        if idx == len(all_3D_points)-1 or not self.env.optimize_psi:
                            return True
                        for non_foldable_idx in range(idx, max_fold_angle_index-1, -1):
                            self.env.triangle_inequality_holds[non_foldable_idx] = False
                        return False
        return False

    def tri_check(self,f_a,f_b,f_a_arr,f_b_arr):
        f_a_arr = f_a_arr.reshape((3,3))
        f_b_arr = f_b_arr.reshape((3,3))
        if sum([a==b for a in f_a for b in f_b])>=2:
            return False # adjacenct or equal
        else:
            return NoDivTriTriIsect(
                    f_a_arr[0,:],f_a_arr[1,:],f_a_arr[2,:],  # face a vertices
                    f_b_arr[0,:],f_b_arr[1,:],f_b_arr[2,:])   # face b vertices

    def get_final_transform(self):
        vec_arr = np.array(self.env.points_2D)[self.surrounding_vertices]-np.array(self.coords)
        surr_vecs = np.hstack((vec_arr,np.zeros(len(self.surrounding_vertices)).reshape(-1,1)))
        surr_vecs_1 = np.roll(surr_vecs,-1,axis=0)
        # === compute sector angles for first unit
        sec_angles = np.array(planar_angle(surr_vecs,surr_vecs_1))[:-1]
        alpha_arr = np.stack((sec_angles,)*len(self.env.psi_opt),axis=0)
        # === collect dihedral angles from vertex to be extended
        rho_arr = self.dihedral_angles.copy()[:,:-1]
        # === resulting first unit
        final_transform = []
        reverse_orient = Rz(np.pi)
        s = np.roll(np.array(alpha_arr),-1)
        d = np.roll(rho_arr,-1)
        d = np.insert(d[:,:-1],0,0,axis=1)
        for i in range(len(self.env.psi_opt)):
            Mffw = Rz(0)
            for a,r in zip(alpha_arr[i],rho_arr[i]):
                Mffw = Mffw.dot(Rx(r).dot(Rz(a)))
            new_rot = self.init_rot[i].dot(Mffw)
            final_transform.append(new_rot)
        return final_transform
