import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.special import fresnel
import csv
import os


# --- Core Contact Class ---
class WheelRailSystem:
    def __init__(self, wheel_center, wheel_quat, V, omega, wheel_type="right"):
        self.wheel_center = np.array(wheel_center)
        self.V = np.array(V)
        self.omega = np.array(omega)
        self.wheel_quat = np.array(wheel_quat)
        self.wheel_type = wheel_type
        self.rot = R.from_quat(wheel_quat)  

        # Wheel profile (mm)
        self.WHEEL_POINTS_X = np.array([39.959, 33.543, 30.832, 25.252,
                                        15.553, -68.447, -71.645, -75.008, -75.947])+7.943
        self.WHEEL_POINTS_R = np.abs(np.array([-475.088, -469.879, -462.389,
                                               -453.968, -451.327, -447.130, -446.470, -444.859, -442.370]))

        if self.wheel_type == "left":
            self.WHEEL_POINTS_X = -self.WHEEL_POINTS_X

        # Rail profile (mm)
        self.RAIL_POINTS_X = np.array([-33.632, -34.622, -35, -35, -31.523, -23, 0,
                                       23, 31.523, 35, 35, 34.622, 33.632])
        self.RAIL_POINTS_Y = np.array([109.244, 109.972, 111.142, 126.2, 135.002, 139.114,
                                       140, 139.114, 135.002, 126.2, 111.142, 109.972, 109.244])-140
        self.create_path()
        self.create_segment_plane()  # Μόνο ένα plane για όλη τη διαδρομή

    def create_path(self):
        """Διαδρομή μόνο ευθεία, χωρίς banking/clothoid/arc."""
        L_straight = 300000
        N_straight = 200
        gauge = 1531.374
        shift = np.array([gauge / 2, 140, 0])
        z1 = np.linspace(0, L_straight, N_straight)
        x1 = np.zeros_like(z1)
        y1 = np.zeros_like(z1)
        r1 = np.column_stack([x1, y1, z1])
        self.path = r1 + shift
        return self.path

    def get_rotation_along_path(self, arc_length):
        ds = np.array([0, 30000])
        deg = np.array([0, 0])
        angle = np.interp(arc_length, ds, deg, left=deg[0], right=deg[-1])
        return angle

    def get_rail_profile_plane_intersection_fast_vec(self, plane, window=20):
        profile_points = np.stack([self.RAIL_POINTS_X, self.RAIL_POINTS_Y], axis=1)  # (N,2)
        path = self.path  # (P,3)
        N_profile = len(profile_points)
        d_path = np.dot(path - plane['point'], plane['normal'])
        idx_min = np.argmin(np.abs(d_path))
        idx_start = max(idx_min - window, 0)
        idx_end = min(idx_min + window, len(path) - 2)
        seg_indices = np.arange(idx_start, idx_end+1)
        M = len(seg_indices)
        baseA = path[seg_indices]
        baseB = path[seg_indices + 1]
        tangentA = baseB - baseA
        norms = np.linalg.norm(tangentA, axis=1, keepdims=True)
        tangentA = np.where(norms > 1e-12, tangentA/norms, np.array([0,0,1]))
        global_y = np.array([0,1,0])
        global_x = np.array([1,0,0])
        projs = np.sum(global_y * tangentA, axis=1, keepdims=True)
        yA = global_y - projs * tangentA
        norms_yA = np.linalg.norm(yA, axis=1, keepdims=True)
        smalls = norms_yA < 1e-6
        projs_x = np.sum(global_x * tangentA, axis=1, keepdims=True)
        yA = np.where(smalls, global_x - projs_x * tangentA, yA)
        norms_yA = np.linalg.norm(yA, axis=1, keepdims=True)
        yA = yA / (norms_yA + 1e-12)
        xA = np.cross(yA, tangentA)
        arc_lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        angles_degA = np.zeros(M)  # πάντα 0 μοίρες (ευθεία)
        angles_radA = np.deg2rad(angles_degA)
        cos_a, sin_a = np.cos(angles_radA), np.sin(angles_radA)
        px = profile_points[:, 0]
        py = profile_points[:, 1]
        pxA = px[:, None] * cos_a[None, :] - py[:, None] * sin_a[None, :]
        pyA = px[:, None] * sin_a[None, :] + py[:, None] * cos_a[None, :]
        ptA = baseA[None, :, :] + pxA[:, :, None]*xA[None, :, :] + pyA[:, :, None]*yA[None, :, :]
        ptB = baseB[None, :, :] + pxA[:, :, None]*xA[None, :, :] + pyA[:, :, None]*yA[None, :, :]  # ίδια γωνία
        dA = np.dot(ptA - plane['point'], plane['normal'])
        dB = np.dot(ptB - plane['point'], plane['normal'])
        crossing_mask = (dA * dB < 0)
        t = dA / (dA - dB + 1e-12)
        t = np.where(crossing_mask, t, 0)
        P = ptA + t[:, :, None] * (ptB - ptA)
        abs_dA = np.abs(dA)
        abs_dB = np.abs(dB)
        min_dA_idx = np.argmin(abs_dA, axis=1)
        min_dB_idx = np.argmin(abs_dB, axis=1)
        fallback_A = ptA[np.arange(N_profile), min_dA_idx]
        fallback_B = ptB[np.arange(N_profile), min_dB_idx]
        use_A = abs_dA[np.arange(N_profile), min_dA_idx] < abs_dB[np.arange(N_profile), min_dB_idx]
        fallback_pts = np.where(use_A[:, None], fallback_A, fallback_B)
        has_crossing = np.any(crossing_mask, axis=1)
        crossing_idx = np.argmax(crossing_mask, axis=1)
        crossing_pts = P[np.arange(N_profile), crossing_idx]
        result = np.where(has_crossing[:, None], crossing_pts, fallback_pts)
        return result

    def find_intersection_with_path(self, plane):
        plane_normal = plane['normal']
        plane_point = plane['point']
        for i in range(len(self.path) - 1):
            A = self.path[i]
            B = self.path[i+1]
            dA = np.dot(plane_normal, A - plane_point)
            dB = np.dot(plane_normal, B - plane_point)
            if dA * dB < 0:
                t = dA / (dA - dB)
                return A + t * (B - A)
        distances = np.abs(np.dot(self.path - plane_point, plane_normal))
        return self.path[np.argmin(distances)]

    def get_plane_local_axes(self, plane):
        origin = plane['point']
        x_axis = plane['x_axis']
        y_axis = plane['y_axis']
        z_axis = plane['z_axis']
        return origin, x_axis, y_axis, z_axis
        
    def create_segment_plane(self):
        """Δημιουργεί μόνο ένα plane για ολόκληρη την ευθεία."""
        midpoint = self.path[len(self.path)//2]
        self.segment_plane = {
            'point': midpoint,
            'normal': np.array([0, 1, 0]),  # Κατακόρυφο
            'segment_points': [self.path[0], midpoint, self.path[-1]],
            'segment_type': 'straight'
        }

    def get_closest_segment_plane(self, point):
        """Βρίσκει το πλησιέστερο plane από τα segment planes"""
        min_dist = float('inf')
        closest_plane = None
        
        for plane in self.segment_planes:
            # Υπολογισμός απόστασης από το επίπεδο
            dist = np.abs(np.dot(plane['normal'], point - plane['point']))
            if dist < min_dist:
                min_dist = dist
                closest_plane = plane
                
        return closest_plane
    
    def get_first_plane(self):
        # Πάντα επιστρέφει το μοναδικό plane της ευθείας
        origin = self.wheel_center
        x_axis = self.rot.apply([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        segment_normal = self.segment_plane['normal']
        projection_on_x = np.dot(segment_normal, x_axis) * x_axis
        y_axis = segment_normal - projection_on_x
        y_axis /= (np.linalg.norm(y_axis) + 1e-12)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        contact_plane = {
            'point': origin,
            'normal': z_axis,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'segment_type': 'straight'
        }
        return [contact_plane]

    def closest_point_on_path_continuous(self, point):
        min_dist = float('inf')
        closest_pt = None
        idx_seg = -1
        for i in range(len(self.path)-1):
            A = self.path[i]
            B = self.path[i+1]
            AB = B - A
            t = np.dot(point - A, AB) / np.dot(AB, AB)
            t = np.clip(t, 0, 1)
            proj = A + t * AB
            dist = np.linalg.norm(point - proj)
            if dist < min_dist:
                min_dist = dist
                closest_pt = proj
                idx_seg = i
        return closest_pt, idx_seg

    def wheel_profile_plane_intersection_exact(self, plane):
        rot = R.from_quat(self.wheel_quat)
        normal = plane['normal']
        point_on_plane = plane['point']
        centers_local = np.column_stack([
            self.WHEEL_POINTS_X, 
            np.zeros_like(self.WHEEL_POINTS_X), 
            np.zeros_like(self.WHEEL_POINTS_X)
        ])  # (N,3)
        centers_global = rot.apply(centers_local) + self.wheel_center  # (N,3)
        v_cos = rot.apply(np.tile([0, 1, 0], (len(centers_local), 1)))  # (N,3)
        v_sin = rot.apply(np.tile([0, 0, 1], (len(centers_local), 1)))  # (N,3)
        dc = np.dot(centers_global - point_on_plane, normal)  # (N,)
        a = np.dot(v_cos, normal) * self.WHEEL_POINTS_R  # (N,)
        b = np.dot(v_sin, normal) * self.WHEEL_POINTS_R  # (N,)
        r = np.hypot(a, b)  # (N,)
        valid = (r > 1e-12) & (np.abs(-dc / r) <= 1)  # (N,)
        rhs = -dc[valid] / r[valid]  # (M,)
        phi = np.arctan2(b[valid], a[valid])  # (M,)
        theta = np.arccos(rhs) + phi  # (M,)
        x_valid = self.WHEEL_POINTS_X[valid]  # (M,)
        R_valid = self.WHEEL_POINTS_R[valid]  # (M,)
        points_local = np.column_stack([
            x_valid,
            R_valid * np.cos(theta),
            R_valid * np.sin(theta)
        ])  # (M,3)
        points_global = rot.apply(points_local) + self.wheel_center  # (M,3)
        valid_y = points_global[:, 1] < self.wheel_center[1]  # (M,)
        return points_global[valid_y] if np.any(valid_y) else None
def closest_point_on_line_to_path(line_origin, line_dir, path):
    # line_dir should be normalized
    min_dist = float('inf')
    best_pt = None
    for pt in path:
        # Προβολή του pt στη γραμμή
        t = np.dot(pt - line_origin, line_dir)
        proj = line_origin + t * line_dir
        dist = np.linalg.norm(pt - proj)
        if dist < min_dist:
            min_dist = dist
            best_pt = pt
    return best_pt
# --- Utility functions ---
def project_to_plane(points, origin, x_axis, y_axis):
    rel = points - origin
    x_local = np.dot(rel, x_axis)
    y_local = np.dot(rel, y_axis)
    return np.column_stack([x_local, y_local])

def sample_points_along_polyline(polyline, samples_per_segment=20):
    points = []
    for i in range(len(polyline)-1):
        p0 = polyline[i]
        p1 = polyline[i+1]
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            pt = (1-t)*p0 + t*p1
            points.append(pt)
    points.append(polyline[-1])
    return np.array(points)

def rail_polygon_n_lines(profile2d):
    return np.vstack([profile2d, profile2d[0]])

def wheel_polygon_contact_segments_multi(wheel_2d, poly_2d):
    path = Path(poly_2d)
    n = len(wheel_2d)
    segments = []
    in_mask = path.contains_points(wheel_2d)
    for i in range(n-1):
        A, B = wheel_2d[i], wheel_2d[i+1]
        in_A = in_mask[i]
        in_B = in_mask[i+1]
        intersections = []
        for j in range(len(poly_2d)-1):
            Q1, Q2 = poly_2d[j], poly_2d[j+1]
            pt = segment_intersection(A, B, Q1, Q2)
            if pt is not None:
                intersections.append(pt)
        points = []
        if in_A:
            points.append(A)
        for pt in intersections:
            points.append(pt)
        if in_B:
            points.append(B)
        if len(points) >= 2:
            points = np.array(points)
            dists = np.linalg.norm(points - A, axis=1)
            idx = np.argsort(dists)
            sorted_points = points[idx]
            for k in range(len(sorted_points)-1):
                mid = 0.5*(sorted_points[k] + sorted_points[k+1])
                if path.contains_point(mid):
                    segments.append([sorted_points[k], sorted_points[k+1]])
        elif in_A and in_B:
            segments.append([A, B])
    all_pts = []
    for seg in segments:
        all_pts.append(tuple(map(tuple, seg)))
    polylines = []
    while all_pts:
        start, end = all_pts.pop(0)
        poly = [start, end]
        changed = True
        while changed:
            changed = False
            for i, (s, e) in enumerate(all_pts):
                if np.allclose(poly[-1], s):
                    poly.append(e)
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[-1], e):
                    poly.append(s)
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[0], s):
                    poly = [e] + poly
                    all_pts.pop(i)
                    changed = True
                    break
                elif np.allclose(poly[0], e):
                    poly = [s] + poly
                    all_pts.pop(i)
                    changed = True
                    break
        polylines.append(np.array(poly))
    return polylines

def segment_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if np.abs(denom) < 1e-12:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    def on_segment(a, b, p):
        return min(a[0], b[0])-1e-10 <= p[0] <= max(a[0], b[0])+1e-10 and min(a[1], b[1])-1e-10 <= p[1] <= max(a[1], b[1])+1e-10
    pt = np.array([px, py])
    if on_segment(p1, p2, pt) and on_segment(q1, q2, pt):
        return pt
    return None

def find_max_min_distance_vector_dense(wheel_poly, rail_poly, samples_per_segment=20):
    sampled_wheel = sample_points_along_polyline(wheel_poly, samples_per_segment)
    seg_a = rail_poly[:-1]
    seg_b = rail_poly[1:]
    pa = sampled_wheel[:, None, :]
    ba = seg_b - seg_a
    ba_len2 = np.sum(ba**2, axis=1)
    ba_len2 = np.where(ba_len2 == 0, 1, ba_len2)
    pa_minus_a = pa - seg_a
    t = np.sum(pa_minus_a * ba, axis=2) / ba_len2
    t = np.clip(t, 0, 1)
    projections = seg_a + t[..., None]*ba
    dists = np.linalg.norm(pa - projections, axis=2)
    min_idx = np.argmin(dists, axis=1)
    min_distances = dists[np.arange(len(sampled_wheel)), min_idx]
    closest_rail_pts = projections[np.arange(len(sampled_wheel)), min_idx]
    vectors = closest_rail_pts - sampled_wheel
    max_idx = np.argmax(min_distances)
    max_min_dist = min_distances[max_idx]
    contact_vec = vectors[max_idx]
    wheel_pt = sampled_wheel[max_idx]
    rail_pt = closest_rail_pts[max_idx]
    return max_min_dist, contact_vec, wheel_pt, rail_pt

def split_polyline_by_length(polyline, max_length):
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(seg_lengths)
    if total_length <= max_length:
        return [polyline]
    cum_length = np.cumsum(seg_lengths)
    split_idx = np.searchsorted(cum_length, total_length / 2)
    if split_idx == 0:
        return [polyline]
    prev_len = cum_length[split_idx-1] if split_idx > 0 else 0
    remain = (total_length/2 - prev_len) / seg_lengths[split_idx]
    split_point = polyline[split_idx] * (1-remain) + polyline[split_idx+1] * remain
    poly1 = np.vstack([polyline[:split_idx+1], split_point])
    poly2 = np.vstack([split_point, polyline[split_idx+1:]])
    return [poly1, poly2]

def softplus(x, beta=100.0):
    return (1.0 / beta) * np.log(1 + np.exp(beta * x))
  
csv_file = None
csv_writer = None
first_call = True
csv_friction_file = None
csv_friction_writer = None
first_friction_call = True

# --- MAIN GFOSUB FUNCTION ---
def GFOSUB(id, time_in, par, npar, dflag, iflag, return_plot_data=False):
    global csv_file, csv_writer, first_call
    global csv_friction_file, csv_friction_writer, first_friction_call

    # Άνοιγμα αρχείου CSV στο πρώτο call
    if first_call:
        csv_file = open('wheel_data_left_front_new_eutheia.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Time', 'dx', 'dy', 'dz', 'yaw', 'pitch', 'roll', 
                            'vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
        first_call = False

    # Άνοιγμα friction CSV στο πρώτο call
    if first_friction_call:
        csv_friction_file = open('friction_data_left_front_new_eutheia.csv', 'w', newline='')
        csv_friction_writer = csv.writer(csv_friction_file)
        csv_friction_writer.writerow([
            'Time', 'dz', 
            'Fx_total', 'Fy_total', 'Fz_total', 
            'Ff_x', 'Ff_y', 'Ff_z', 
            'ContactX', 'ContactY', 'ContactZ',
            'Vx', 'Vy', 'Vz',
            'v_tangent_mag_avg', 'v_normal_mag',
            'max_penetration', 'avg_penetration',
            'contact_area', 'is_Ff_z_pos', 'is_Ff_x_pos', 'is_sliding', 'friction_coeff'
        ])
        first_friction_call = False

    [dx, _] = py_sysfnc("DX", [par[0], 30101010])
    [dy, _] = py_sysfnc("DY", [par[0], 30101010])
    [dz, _] = py_sysfnc("DZ", [par[0], 30101010])
    [vx, _] = py_sysfnc("VX", [par[0], 30101010])
    [vy, _] = py_sysfnc("VY", [par[0], 30101010])
    [vz, _] = py_sysfnc("VZ", [par[0], 30101010])
    [yaw, _] = py_sysfnc("YAW", [par[0], 30101010])
    [pitch, _] = py_sysfnc("PITCH", [par[0], 30101010])
    [roll, _] = py_sysfnc("ROLL", [par[0], 30101010])
    [wx, _] = py_sysfnc("WX", [par[0], 30101010])
    [wy, _] = py_sysfnc("WY", [par[0], 30101010])
    [wz, _] = py_sysfnc("WZ", [par[0], 30101010])

    # Γράψε τις τιμές στο CSV
    csv_writer.writerow([time_in, dx, dy, dz, yaw, pitch, roll, 
                        vx, vy, vz, wx, wy, wz])

    print("\n--- Left Front Wheel State ---")
    print(f"Position:    dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    print(f"Orientation: yaw={yaw:.4f}, pitch={pitch:.4f}, roll={roll:.4f}")
    print(f"Velocities:  vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}")
    print(f"Angular Vel: wx={wx:.4f}, wy={wy:.4f}, wz={wz:.4f}")

    wheel_center = [dx, dy, dz]
    rot = R.from_euler('zyx', [yaw, pitch, roll])
    wheel_quat = rot.as_quat()
    V = [vx, vy, vz]
    omega = [wx, wy, wz]
    wheel_type = "left"
    system = WheelRailSystem(wheel_center, wheel_quat, V, omega, wheel_type=wheel_type)
    plane = system.get_first_plane()[0]
    origin, x_axis, y_axis, z_axis = system.get_plane_local_axes(plane)

    sign_x = 1 if wheel_type == 'left' else -1
    x_axis = sign_x * x_axis
    wheel_2d = np.column_stack([sign_x * system.WHEEL_POINTS_X, -system.WHEEL_POINTS_R])
    wheel_profile_3d = origin + np.outer(wheel_2d[:, 0], x_axis) + np.outer(wheel_2d[:, 1], y_axis)
    rail_profile_3d = system.get_rail_profile_plane_intersection_fast_vec(plane)
    rail_2d = project_to_plane(rail_profile_3d, origin, x_axis, y_axis)
    rail_poly = rail_polygon_n_lines(rail_2d)
    wheel_contact_polylines = wheel_polygon_contact_segments_multi(wheel_2d, rail_poly)

    max_length = 15.0  # mm
    max_min_info = []
    for poly in wheel_contact_polylines:
        subpolys = split_polyline_by_length(poly, max_length)
        for subpoly in subpolys:
            if len(subpoly) > 1:
                max_min_dist, contact_vec, wheel_pt, rail_pt = find_max_min_distance_vector_dense(subpoly, rail_poly, samples_per_segment=30)
                max_min_info.append({
                    "max_min_dist": max_min_dist,
                    "vec": contact_vec,
                    "pt_on_poly": wheel_pt,
                    "pt_on_polygon": rail_pt,
                    "axes": (origin, x_axis, y_axis, z_axis)
                })

    # --- Υπολογισμός δυνάμεων και ροπών επαφής στο κύριο plane ---
    K = 1000000
    n = 1.5
    C_max = 100  # Μέγιστη απόσβεση
    penetration_limit = 0.1  # mm (όριο για πλήρη απόσβεση)
    max_penetration_allowed = 5# mm

    max_damping_velocity = 10.0 / 0.001  # mm/s
    mu_dynamic = 0.2  # Συντελεστής δυναμικής τριβής
    v_threshold = 20 # Νέα τιμή για γραμμική τριβή (1 μm/s)

    wheel_center_np = np.array(wheel_center)
    F_total_main = np.zeros(3)
    M_total_main = np.zeros(3)
    forces_plot_main = []

    for info in max_min_info:
        penetration_depth = abs(info["max_min_dist"])
        penetration_depth = softplus(penetration_depth, beta=20)
        if penetration_depth > max_penetration_allowed:
            penetration_depth = max_penetration_allowed

        origin, x_axis, y_axis, z_axis = info["axes"]
        Pmax2d = info["pt_on_poly"]
        Qrail2d = info["pt_on_polygon"]
        Pmax = origin + Pmax2d[0]*x_axis + Pmax2d[1]*y_axis
        Qrail = origin + Qrail2d[0]*x_axis + Qrail2d[1]*y_axis

        penetration_dir = Pmax - Qrail
        penetration_dir_norm = np.linalg.norm(penetration_dir)
        if penetration_dir_norm < 1e-12:
            continue
        penetration_dir = penetration_dir / penetration_dir_norm

        Fx = np.dot(penetration_dir, x_axis)
        Fy = np.dot(penetration_dir, y_axis)
        Fz = np.dot(penetration_dir, z_axis)
        penetration_dir_mod = Fx * x_axis + Fy * y_axis + Fz * z_axis
        penetration_dir_mod_norm = np.linalg.norm(penetration_dir_mod)
        if penetration_dir_mod_norm < 1e-12:
            continue
        penetration_dir_mod = penetration_dir_mod / penetration_dir_mod_norm

        # Elastic force
        F_elastic = -K * (penetration_depth ** n) * penetration_dir_mod
        
        # Damping force (με μεταβλητό C)
        v_point = np.array(V) + np.cross(np.array(omega), (Pmax - wheel_center_np))
        elastic_dir = -penetration_dir_mod
        v_proj_elastic = np.dot(v_point, elastic_dir)
        
        if penetration_depth > 1e-6 and v_proj_elastic < -1e-6:
            # Υπολογισμός C με γραμμική παρεμβολή
            C_effective = C_max * min(penetration_depth / penetration_limit, 1.0)
            v_damp_clipped = min(abs(v_proj_elastic), max_damping_velocity) * np.sign(v_proj_elastic)
            F_damp = C_effective * abs(v_damp_clipped) * elastic_dir
        else:
            F_damp = np.zeros(3)
        
        F_normal = np.linalg.norm(F_elastic + F_damp)

        # Tangential velocity and friction (παραμένει ίδιο)
        v_tan = v_point - np.dot(v_point, penetration_dir_mod) * penetration_dir_mod
        v_tan_norm = np.linalg.norm(v_tan)
        v_tan_safe = max(v_tan_norm, 1e-8)
        tangent_unit_vector = v_tan / v_tan_safe

        # Linear friction model (0 to mu_dynamic)
        def compute_friction_coefficient(v_tan_norm, mu_dynamic=0.2, v_d=1e-3):
            safe_vd = max(v_d, 1e-12)
            clipped_v = max(v_tan_norm, 0.0)
            return mu_dynamic * min(clipped_v / safe_vd, 1.0)

        mu = compute_friction_coefficient(v_tan_norm, mu_dynamic, v_threshold)
        F_friction = -mu * F_normal * tangent_unit_vector
        
        # Μετατροπή δυνάμεων σε τοπικούς άξονες (παραμένει ίδιο)
        F_elastic_local = np.array([np.dot(F_elastic, x_axis), np.dot(F_elastic, y_axis), np.dot(F_elastic, z_axis)])
        F_damp_local = np.array([np.dot(F_damp, x_axis), np.dot(F_damp, y_axis), np.dot(F_damp, z_axis)])
        
        # Moments (παραμένει ίδιο)
        r_vec = Pmax - wheel_center_np
        M_elastic = np.cross(r_vec, F_elastic)
        M_damp = np.cross(r_vec, F_damp)
        M_friction = np.cross(r_vec, F_friction)
        print(f"F_elastic: [{F_elastic[0]:.6f}, {F_elastic[1]:.6f}, {F_elastic[2]:.6f}]")
        print(f"F_damp:    [{F_damp[0]:.6f}, {F_damp[1]:.6f}, {F_damp[2]:.6f}]")
        print(f"F_friction: [{F_friction[0]:.6f}, {F_friction[1]:.6f}, {F_friction[2]:.6f}]")
        F_total_main += F_elastic + F_damp + F_friction
        M_total_main += M_elastic + M_damp + M_friction

        forces_plot_main.append({
            'pt': Pmax,
            'F_elastic': F_elastic,
            'F_damp': F_damp,
            'F_friction': F_friction,
            'penetration_dir_mod': penetration_dir_mod,
            'penetration_depth': penetration_depth,
            'tangent_unit_vector': tangent_unit_vector
        })



    F_total = np.zeros(3)
    M_total = np.zeros(3)
    for f in forces_plot_main:
        F = f['F_elastic'] + f['F_damp'] + f['F_friction']
        F_total += F
        r_vec = f['pt'] - np.array(wheel_center)
        M_total += np.cross(r_vec, F)

    # --- Υπολογισμοί friction csv ---
    F_friction_total = np.zeros(3)
    contact_points = []
    velocities = []
    penetrations = []
    v_tangent_mags = []
    v_normal_mags = []

    for f in forces_plot_main:
        F_friction_total += f['F_friction']
        contact_points.append(f['pt'])
        v_contact = np.array(V) + np.cross(np.array(omega), f['pt'] - wheel_center_np)
        velocities.append(v_contact)
        v_tan = v_contact - np.dot(v_contact, f['penetration_dir_mod']) * f['penetration_dir_mod']
        v_tangent_mags.append(np.linalg.norm(v_tan))
        v_normal_mags.append(np.abs(np.dot(v_contact, f['penetration_dir_mod'])))
        penetrations.append(f['penetration_depth'])

    if contact_points:
        avg_contact_point = np.mean(np.vstack(contact_points), axis=0)
        avg_velocity = np.mean(np.vstack(velocities), axis=0)
        v_tangent_mag_avg = np.mean(v_tangent_mags)
        v_normal_mag = np.mean(v_normal_mags)
        max_penetration = np.max(penetrations)
        avg_penetration = np.mean(penetrations)
    else:
        avg_contact_point = np.zeros(3)
        avg_velocity = np.zeros(3)
        v_tangent_mag_avg = 0
        v_normal_mag = 0
        max_penetration = 0
        avg_penetration = 0

    contact_area = sum(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)) for poly in wheel_contact_polylines)
    is_sliding = int(any(v > 1e-8 for v in v_tangent_mags))
    friction_coeff = np.mean([compute_friction_coefficient(v) for v in v_tangent_mags]) if v_tangent_mags else 0

    # --- Κριτήριο: ΜΟΝΟ ΑΝ dz < 30000 ΚΑΙ F_friction_total[2] > 0 ---
    if dz < 30000 and F_friction_total[2] > 0:
        csv_friction_writer.writerow([
            time_in,
            dz,
            F_total[0], F_total[1], F_total[2],
            F_friction_total[0], F_friction_total[1], F_friction_total[2],
            avg_contact_point[0], avg_contact_point[1], avg_contact_point[2],
            avg_velocity[0], avg_velocity[1], avg_velocity[2],
            v_tangent_mag_avg,
            v_normal_mag,
            max_penetration,
            avg_penetration,
            contact_area,
            int(F_friction_total[2] > 0),
            int(F_friction_total[0] > 0),
            is_sliding,
            friction_coeff
        ])

    if return_plot_data:
        return {
            "forces_main_filtered": forces_plot_main,
            "F_total": F_total,
            "M_total": M_total,
            "forces_plot_main": forces_plot_main,
        }
    else:
        result = list(F_total) + list(M_total)
        return result
        
  # Συνάρτηση για να κλείσει το αρχείο όταν τελειώσει η προσομοίωση
def close_csv_file():
    global csv_file
    if csv_file is not None:
        csv_file.close()
        csv_file = None
def close_csv_friction_file():
    global csv_friction_file
    if csv_friction_file is not None:
        csv_friction_file.close()
        csv_friction_file = None