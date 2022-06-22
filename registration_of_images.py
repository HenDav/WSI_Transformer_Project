import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from tqdm import tqdm
from scipy.interpolate import griddata, interp2d
import os
from random import randint
import matplotlib.pyplot as plt

import pandas as pd



def get_convex_triangle_area(delanuay, simplex_match):
	areas = []
	for simplex_idx in range(simplex_match.shape[0]):
		points = delanuay.simplices[simplex_match[simplex_idx]]

		# Get triangle point coordinates:
		point_coordinates = np.zeros((3, 2))
		for i, point_number in enumerate(points):
			point_coordinates[i] = delanuay.points[point_number]

		# Compute triangle side length:
		l0 = np.sqrt(((point_coordinates[0] - point_coordinates[1]) ** 2).sum())
		l1 = np.sqrt(((point_coordinates[1] - point_coordinates[2]) ** 2).sum())
		l2 = np.sqrt(((point_coordinates[0] - point_coordinates[2]) ** 2).sum())

		p = (l0 + l1 + l2) / 2
		area = np.sqrt(p * (p - l0) * (p - l1) * (p - l2))

		areas.append(area)

	return np.array(areas)


def get_closest_mid_point_index(pix_original_location, simplex_areas):
	global mid_points
	EPS = 10
	differences = mid_points - pix_original_location
	distances = np.sqrt((differences ** 2).sum(1))
	distances /= (simplex_areas + EPS * simplex_areas.mean())  # Normalize distance by triangle area
	closest_simplex = np.argmin(distances)

	return closest_simplex


def get_closest_gils_value(pix_original_location, inv_simplices_values):
	pix_original_location = np.hstack((pix_original_location, 1)).reshape(3, 1)
	#distances = abs(np.matmul(inv_simplices_values, pix_original_location)).squeeze(2).mean(1)
	distances = np.linalg.norm(np.matmul(inv_simplices_values, pix_original_location).squeeze(2), axis=1)
	closest_simplex = np.argmin(distances)
	return closest_simplex


def get_closest_combined(pix_original_location, inv_simplices_values, alpha):
	global mid_points
	differences = mid_points - pix_original_location
	true_distances = np.sqrt((differences ** 2).sum(1))
	pix_original_location = np.hstack((pix_original_location, 1)).reshape(3, 1)
	gils_distances = abs(np.matmul(inv_simplices_values, pix_original_location)).squeeze(2).mean(1)

	total_distances = alpha * true_distances + (1 - alpha) * gils_distances
	closest_simplex = np.argmin(total_distances)
	return closest_simplex


def get_closest_simplex(pix_original_location, simplices_inv_values, simplex_areas, alpha, is_grad_metric=False):
	global mid_points
	true_distances = np.sqrt(((mid_points - pix_original_location) ** 2).sum(1))
	pix_original_location = np.hstack((pix_original_location, 1)).reshape(3, 1)
	gils_distances = abs(np.matmul(inv_simplices_values, pix_original_location)).squeeze(2).mean(1)

	total_distances = alpha * true_distances + (1 - alpha) * gils_distances
	closest_simplex = np.argmin(total_distances)
	return closest_simplex


def compute_mid_points_for_convex_hull(delanuay_triangles_data):
	all_points = delanuay_triangles_data.points
	convex_hull_lines = delanuay_triangles_data.convex_hull

	mid_points = []
	match_array = []

	for i in range(convex_hull_lines.shape[0]):
		current_line_points = np.array([all_points[convex_hull_lines[i][0]], all_points[convex_hull_lines[i][1]]])
		mid_point = current_line_points.sum(0) / 2
		mid_points.append(mid_point)

		in_simplex = HE_tri.find_simplex(mid_point)
		match_array.append(in_simplex)

	mid_points = np.array(mid_points)
	match_array = np.array(match_array)

	return mid_points, match_array


def compute_CG_points_for_simplices(delanuay_triangles_data):
	all_points = delanuay_triangles_data.points
	simplices_lines = delanuay_triangles_data.simplices

	CG_points = []
	match_array = []

	for i in range(simplices_lines.shape[0]):
		current_line_points = np.array([all_points[simplices_lines[i][0]],
										all_points[simplices_lines[i][1]],
										all_points[simplices_lines[i][2]]])
		CG = current_line_points.sum(0) / 3
		CG_points.append(CG)

		in_simplex = HE_tri.find_simplex(CG)
		match_array.append(in_simplex)

	CG_points = np.array(CG_points)
	match_array = np.array(match_array)

	return CG_points, match_array


def compute_Gils_value_simplices(delanuay_triangles_data):
	all_points = delanuay_triangles_data.points
	simplices_lines = delanuay_triangles_data.simplices

	inv_simplices_values = []
	match_array = []

	for i in range(simplices_lines.shape[0]):
		current_triangle_points = np.array([all_points[simplices_lines[i][0]],
											all_points[simplices_lines[i][1]],
											all_points[simplices_lines[i][2]]])

		CG = current_triangle_points.sum(0) / 3
		in_simplex = HE_tri.find_simplex(CG)
		match_array.append(in_simplex)

		current_triangle_points = np.transpose(np.hstack((current_triangle_points, np.array(([1], [1], [1])))))
		current_triangle_points_inverted = np.linalg.inv(current_triangle_points)
		inv_simplices_values.append(current_triangle_points_inverted)


	inv_simplices_values = np.array(inv_simplices_values)
	match_array = np.array(match_array)

	return inv_simplices_values, match_array


def interpolate_image(img2, grid_x, grid_y):
	H, W, _ = img2.shape
	new_img_H, new_img_W = grid_x.shape
	new_img = np.zeros((new_img_H, new_img_W, 3), dtype=np.uint8)

	grid_x /= W
	grid_y /= H
	x = np.linspace(0, 1, W)  #np.linspace(0, W - 1, W)
	y = np.linspace(0, 1, H)  #np.linspace(0, H - 1, H)
	xv, yv = np.meshgrid(x, y)

	x_original = np.reshape(xv, (-1, 1))
	y_original = np.reshape(yv, (-1, 1))

	points = np.hstack((y_original, x_original))

	values_R = np.reshape(img2[:, :, 0], (-1, 1))
	values_G = np.reshape(img2[:, :, 1], (-1, 1))
	values_B = np.reshape(img2[:, :, 2], (-1, 1))

	print('Interpolating R channel')
	R_interpolated = griddata(points=points, values=values_R, xi=(grid_y, grid_x), fill_value=0, method='linear')

	print('Interpolating G channel')
	G_interpolated = griddata(points=points, values=values_G, xi=(grid_y, grid_x), fill_value=0, method='linear')

	print('Interpolating B channel')
	B_interpolated = griddata(points=points, values=values_B, xi=(grid_y, grid_x), fill_value=0, method='linear')

	print('Finished interpolating !')

	R_interpolated = np.reshape(R_interpolated.astype(np.uint8), (H, W))
	G_interpolated = np.reshape(G_interpolated.astype(np.uint8), (H, W))
	B_interpolated = np.reshape(B_interpolated.astype(np.uint8), (H, W))

	new_img[:, :, 0] = R_interpolated
	new_img[:, :, 1] = G_interpolated
	new_img[:, :, 2] = B_interpolated

	return new_img


def create_pixel_location_matrix(img1, simplex_match, simplex_areas, simplex_inverse_values, is_by_gils_values=False):
	global HE_tri
	H, W, _ = img1.shape
	grid_y = (-1) * np.ones((H, W))
	grid_x = (-1) * np.ones((H, W))

	vor_image = np.zeros_like(img1)
	vor_image[:, :, :] = img1[:, :, :]

	colors = [[51, 102, 0],
			  [0, 51, 102],
			  [153, 0, 153],
			  [255, 0, 127],
			  [51, 51, 255],
			  [255, 51, 153],
			  [255, 153, 153],
			  [255, 255, 102],
			  [153, 204, 255],
			  [255, 104, 255]]


	colors_random = [[randint(0, 255) for j in range(3)] for i in range(HE_tri.nsimplex)]

	colors = np.array(colors_random)

	# Original Pixel Locations:
	x = np.linspace(0, W - 1, W)
	y = np.linspace(0, H - 1, H)
	xv, yv = np.meshgrid(x, y)
	xv = xv.reshape((1, -1))
	yv = yv.reshape((1, -1))
	original_coordinates = np.vstack((yv, xv)).transpose()
	in_triangle = HE_tri.find_simplex(original_coordinates)
	closest_index_of_mid_point = get_closest_simplex(original_coordinates, simplex_inverse_values, simplex_areas, alpha=0.1, is_grad_metric=True)


	for _, col in enumerate(tqdm(range(W))):
		for row in range(H):
			pix_original_location = np.array([row, col])
			in_triangle = HE_tri.find_simplex(pix_original_location)

			if in_triangle == -1:  # in case the point is out of all the simplices, we need to compute the closest simplex
				if is_by_gils_values:
					# Compute "Distances":
					#closest_index_of_mid_point = get_closest_gils_value(pix_original_location, simplex_inverse_values)
					closest_index_of_mid_point = get_closest_combined(pix_original_location, simplex_inverse_values, alpha=0.1)

				else:
					closest_index_of_mid_point = get_closest_mid_point_index(pix_original_location, simplex_areas)

				in_triangle = simplex_match[closest_index_of_mid_point]
				vor_image[row, col, :] = colors[np.where(simplex_match == in_triangle)]

			if in_triangle in simplex_match:
				vor_image[row, col, :] = colors[np.where(simplex_match == in_triangle)]

			pix_transformed_coordinates = apply_points_transform(pix_original_location, transform=transform_matrices[in_triangle])
			grid_y[row, col] = pix_transformed_coordinates[0, 0]  # first coordinate is ROW
			grid_x[row, col] = pix_transformed_coordinates[0, 1]  # second coordinate is COLUMN

	draw_delaunay(vor_image, HE_cord, HE_tri.simplices)
	cv2.imshow('voronoi', vor_image)
	cv2.waitKey(0)


	return grid_x, grid_y


'''def zeroize_transforms(transforms, which):
	if which >= len(transforms):
		raise Exception('Number too big')

	all_nums = [num for num in range(len(transforms))]
	all_nums.remove(which)
	for i in all_nums:
		transforms[i] = np.zeros_like(transforms[i ])

	return transforms'''


def pick_and_show_point(event, x, y, flags, params):
	global W, HE_tri, transform_matrices

	if event == cv2.EVENT_LBUTTONDOWN:
		if x <= W:
			in_triangle = HE_tri.find_simplex(np.array([x, y]))

			if in_triangle != -1:
				cv2.circle(img, center=(x, y), radius=7, color=(0, 0, 255), thickness=3)
				cv2.imshow('image2', img)

				img2_coord = apply_points_transform(np.array([x, y]), transform=transform_matrices[in_triangle])
				img2_coord = img2_coord.astype(np.int32)

				cv2.circle(img, center=(img2_coord[0, 0] + W, img2_coord[0, 1]), radius=40, color=(255, 255, 0), thickness=-1)
				cv2.imshow('image2', img)


def apply_points_transform(points, transform):
	if type(points) == list:
		points = np.array(points)

	if len(points.shape) == 1:
		points = points.reshape((1, 2))

	ones = np.ones((points.shape[0], 1))

	points = np.transpose(np.hstack((points, ones)))

	output = np.transpose(np.matmul(transform, points))

	return output


def get_transforms(points_HE, points_IHC, triangles) -> list:
	Transform_Matrices = []
	for tri in triangles:
		p_HE = [points_HE[t] for t in tri]
		p_IHC = [points_IHC[t] for t in tri]

		Transform_Matrices.append(cv2.getAffineTransform(np.float32(p_HE), np.float32(p_IHC)))

	return Transform_Matrices


def rect_contains(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
		return False

	return True


def draw_delaunay(img, points, triangles):
	size = img.shape
	r = (0, 0, size[1], size[0])
	for t in triangles:
		pt1 = (points[t[0]][1], points[t[0]][0])
		pt2 = (points[t[1]][1], points[t[1]][0])
		pt3 = (points[t[2]][1], points[t[2]][0])
		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
			cv2.line(img, pt1, pt2, (255, 0, 0), 5)
			cv2.line(img, pt2, pt3, (255, 0, 0), 5)
			cv2.line(img, pt3, pt1, (255, 0, 0), 5)


def draw_points(img, points, size:int = 15):
	for p in points:
		p = (int(p[1]), int(p[0]))
		cv2.circle(img, center=p, radius=size, color=(0, 0, 255), thickness=-1)


def fix_coordinates(coordinates: list, img_HE, img_IHC) -> list:
	fixed_coordinates_HE, fixed_coordinates_IHC = [], []
	_, width, _ = img_HE.shape
	for cord in coordinates:
		if cord[0] > width:
			fixed_coordinates_IHC.append((cord[1], cord[0] - width))
		else:
			fixed_coordinates_HE.append((cord[1], cord[0]))

	return fixed_coordinates_HE, fixed_coordinates_IHC


def mark_points(coordinates: list, img):
	for cord in coordinates:
		cv2.circle(img, center=cord, radius=15, color=(255, 0, 0), thickness=-1)

	cv2.imshow('image', img)


def click_event(event, x, y, flags, params):
	global coordinates_Left
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(img, center=(x, y), radius=10, color=(255, 0, 0), thickness=3)

		coordinates_Left.append((x,y))
		cv2.imshow('image', img)

	'''
	# checking for right mouse clicks
	if event==cv2.EVENT_RBUTTONDOWN:
		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)
	'''



if __name__=="__main__":

	coordinates_Left = []
	# reading the image

	'''file1 = '/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/19-5229_2_1_e.jpg'
	file2 = '/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/19-5229_2_1_l.jpg'
	HE_cord = [(324, 217), (306, 324), (268, 383), (239, 404), (209, 369), (409, 500), (428, 532), (407, 587),
			   (366, 569), (376, 552), (378, 493), (439, 409), (312, 391), (267, 445), (224, 458), (156, 387), (188, 319),
			   (240, 219), (339, 119), (463, 81), (592, 237), (455, 139), (643, 189), (685, 271), (681, 327), (385, 296),
			   (338, 278), (249, 273), (364, 347), (455, 351), (483, 385), (555, 667), (545, 735), (375, 688), (690, 375),
			   (756, 365), (642, 643), (555, 526), (594, 547), (555, 638), (487, 658), (456, 606), (473, 564), (512, 565)]
	IHC_cord = [(1258, 193), (1228, 305), (1180, 351), (1153, 366), (1131, 339), (1318, 485), (1333, 531), (1307, 578),
				(1268, 545), (1280, 518), (1284, 471), (1351, 404), (1230, 369), (1187, 409), (1132, 418), (1071, 347),
				(1105, 295), (1180, 195), (1264, 121), (1414, 78), (1509, 244), (1401, 138), (1559, 199), (1580, 291),
				(1574, 345), (1305, 282), (1259, 261), (1179, 239), (1285, 325), (1366, 348), (1389, 383), (1422, 666), (1393, 724),
				(1261, 668), (1600, 407), (1666, 396), (1522, 647), (1455, 531), (1479, 546), (1441, 638), (1374, 648),
				(1350, 603), (1372, 555), (1411, 559)]'''

	file1 = '/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_e_TOP.jpg'
	file2 = '/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_l.jpg'


	HE_cord = [[4143, 2773], [3913, 4141], [3427, 4895], [3056, 5164], [2672, 4716], [5230, 6391], [5473, 6800], [5204, 7503], [4680, 7273], [4808, 7056], [4833, 6301], [5613, 5228], [3989, 4998], [3414, 5688], [2864, 5854], [1994, 4946], [2404, 4077], [3069, 2799], [4335, 1521], [5920, 1035], [7570, 3029], [5818, 1776], [8222, 2415], [8759, 3464], [8708, 4179], [4923, 3783], [4322, 3553], [3184, 3489], [4654, 4435], [5818, 4486], [6176, 4921], [7097, 8526], [6969, 9395], [4795, 8794], [8823, 4793], [9667, 4665], [8209, 8219], [7097, 6723], [7596, 6992], [7097, 8155], [6227, 8411], [5831, 7746], [6048, 7209], [6547, 7222]]


	IHC_cord = [[16087, 2467], [15703, 3898], [15089, 4486], [14744, 4678], [14463, 4333], [16854, 6199], [17046, 6787], [16713, 7388], [16215, 6966], [16368, 6621], [16419, 6020], [17276, 5164], [15729, 4716], [15179, 5228], [14476, 5343], [13695, 4435], [14130, 3770], [15089, 2492], [16164, 1546], [18082, 997], [19297, 3119], [17915, 1764], [19936, 2543], [20205, 3719], [20128, 4410], [16688, 3604], [16100, 3336], [15077, 3055], [16432, 4154], [17468, 4448], [17762, 4895], [18184, 8513], [17813, 9254], [16125, 8538], [20460, 5202], [21304, 5062], [19463, 8270], [18606, 6787], [18913, 6979], [18427, 8155], [17570, 8283], [17263, 7708], [17545, 7094], [18043, 7145]]

	#HE_cord = [(322, 206), (313, 395), (482, 385)]
	#IHC_cord = [(1259, 180), (1226, 368), (1387, 392)]
	'''HE_cord = [(341, 114), (162, 374), (547, 732)]
	IHC_cord = [(1273, 111), (1075, 342), (1398, 721)]'''


	# 18-7361_1_7_b.jpg
	#HE_cord = [(270, 238), (86, 664), (191, 885), (22, 1201), (207, 1468), (388, 1371), (362, 1086), (541, 1376), (929, 1023), (1454, 865), (1344, 630), (1281, 258), (1323, 70), (815, 125)]
	#IHC_cord = [(553, 2207), (938, 2441), (1189, 2371), (1542, 2567), (1789, 2376), (1711, 2199), (1389, 2206), (1715, 2060), (1372, 1613), (1299, 1103), (1011, 1193), (675, 1221), (476, 1204), (482, 1676)]



	img1 = cv2.imread(file1, 1)
	location = '/'.join(file1.split('/')[:-1])
	file_name_1 = file1.split('/')[-1]

	'''img2 = cv2.imread(file2, 1)
	file_name_2 = file2.split('/')[-1]
	img2_original = cv2.imread(file2, 1)'''


	'''
	img1 = cv2.imread('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/18-7361_1_7_b.jpg', 1)
	img2 = cv2.imread('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/18-7361_1_7_c.jpg', 1)
	img2_original = cv2.imread('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/18-7361_1_7_c.jpg', 1)
	'''

	'''
	img1 = cv2.imread('/Users/wasserman/Downloads/Dog', 1)
	img2 = cv2.imread('/Users/wasserman/Downloads/Dog', 1)
	HE_cord = [(24, 9), (19, 218), (164, 131)]
	IHC_cord = [(93, 45), (37, 107), (148, 158)]
	'''




	H, W, _ = img1.shape

	if True:
		#img = np.concatenate((img1, img2), axis=1)
		img=img1
		# displaying the image
		cv2.imshow('image', img)

		# setting mouse handler for the image
		# and calling the click_event() function
		cv2.setMouseCallback('image', click_event)

		# wait for a key to be pressed to exit
		cv2.waitKey(0)

		# close the window
		cv2.destroyAllWindows()
		HE_cord, IHC_cord = fix_coordinates(coordinates_Left, img1, img1)
		print(HE_cord)
		print(IHC_cord)

		'''
		mark_points(HE_cord, img1)
		cv2.waitKey(0)
		mark_points(IHC_cord, img2)
		cv2.waitKey(0)
		'''


	#HE_cord = [(240, 264), (657, 88), (1246, 27), (1450, 217), (1378, 381), (1081, 359), (1357, 543), (1052, 910), (872, 1432), (281, 1280), (86, 1121)]
	#IHC_cord = [(2210, 542), (2437, 926), (2558, 1521), (2361, 1759), (2204, 1704), (2199, 1383), (2057, 1718), (1608, 1399), (1113, 1300), (1209, 650), (1397, 460)]

	#HE_cord = [(244, 264), (1367, 540), (865, 1445)]
	#IHC_cord = [(2211, 553), (2040, 1715), (1125, 1267)]



	HE_tri = Delaunay(HE_cord)
	#draw_delaunay(img1, HE_cord, HE_tri.simplices)

	draw_delaunay(img2, IHC_cord, HE_tri.simplices)


	transform_matrices = get_transforms(HE_cord, IHC_cord, HE_tri.simplices)


	#mid_points, simplex_match = compute_mid_points_for_convex_hull(HE_tri)

	# Extrapulations outside the triangles
	mid_points, simplex_match = compute_CG_points_for_simplices(HE_tri)
	areas = get_convex_triangle_area(HE_tri, simplex_match)
	inv_simplices_values, simplex_match = compute_Gils_value_simplices(HE_tri)


	cv2.imwrite(os.path.join(location, 'With_Triangles_' + file_name_1), img1)
	cv2.imwrite(os.path.join(location, 'With_Triangles_' + file_name_2), img2)



	img = np.concatenate((img1, img2), axis=1)

	'''
	cv2.imshow('image2', img)
	cv2.setMouseCallback('image2', pick_and_show_point)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	#x_cords, y_cords = create_pixel_location_matrix(img1, simplex_match, areas)
	#x_cords, y_cords = create_pixel_location_matrix(img1, simplex_match, inv_simplices_values, is_by_gils_values=True)
	x_cords, y_cords = create_pixel_location_matrix(img1=img1,
													simplex_match=simplex_match,
													simplex_areas=areas,
													simplex_inverse_values=inv_simplices_values,
													is_by_gils_values=True)

	if False:  # Save coordinates for interpolation
		X_PD = pd.DataFrame(x_cords)
		Y_PD = pd.DataFrame(y_cords)
		X_PD.to_csv(os.path.join(location, 'X_Cords.csv'))
		Y_PD.to_csv(os.path.join(location, 'Y_Cords.csv'))

	new_img = interpolate_image(img2, x_cords, y_cords)
	cv2.imwrite(os.path.join(location, 'CONVERTED_With_Triangles_' + file_name_2), new_img)


	converted_image = np.concatenate((img, np.concatenate((img1, new_img), axis=1)), axis=0)

	cv2.imshow('Converted_Image', converted_image)
	cv2.waitKey(0)

	print('Done')

