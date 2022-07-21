import openslide
import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import cv2
from PIL import Image


class Tile:
    def __init__(self, center=None, pix_ratio_from_10MAG=None, delaunay=None, Mag10_TileSize: int = 256):
        self.Center = center  # (ROW, COL)
        self.tile_size = Mag10_TileSize
        half_tile = Mag10_TileSize / 2
        self.TopLeft = (center[0] - int(half_tile / pix_ratio_from_10MAG['HE']['H']), center[1] - int(half_tile / pix_ratio_from_10MAG['HE']['W']))
        self.TopRight = (center[0] - int(half_tile / pix_ratio_from_10MAG['HE']['H']), center[1] + int(half_tile / pix_ratio_from_10MAG['HE']['W']))
        self.BottomLeft = (center[0] + int(half_tile / pix_ratio_from_10MAG['HE']['H']), center[1] - int(half_tile / pix_ratio_from_10MAG['HE']['W']))
        self.BottomRight = (center[0] + int(half_tile / pix_ratio_from_10MAG['HE']['H']), center[1] + int(half_tile / pix_ratio_from_10MAG['HE']['W']))
        self.in_simplex = delaunay.find_simplex(self.Center).item()


    def set_IHC_properties(self, transform):
        tile_corners = np.transpose(np.hstack((np.array((self.TopLeft, self.TopRight, self.BottomLeft, self.BottomRight)),
                                               np.ones(4).reshape(4, 1))))
        IHC_tile_corners = np.transpose(np.matmul(transform, tile_corners)).astype(int)

        self.IHC_TopLeft = IHC_tile_corners[0, :]
        self.IHC_TopRight = IHC_tile_corners[1, :]
        self.IHC_BottomLeft = IHC_tile_corners[2, :]
        self.IHC_BottomRight = IHC_tile_corners[3, :]


def create_tiles(centers, pix_ratio_from_10MAG, delaunay):
    return [Tile(center, pix_ratio_from_10MAG, delaunay) for center in centers]


def plot_tiles(img_HE, img_IHC, tiles):
    thickness = 2
    for tile in tiles:
        # Draw HE Tiles
        cv2.rectangle(img_HE, tile.TopLeft[::-1], tile.BottomRight[::-1], color=(255, 0, 0), thickness=thickness)

        # Draw IHC Tiles
        cv2.polylines(img_IHC,
                      [np.array((tile.IHC_TopLeft[::-1],
                       tile.IHC_TopRight[::-1],
                       tile.IHC_BottomRight[::-1],
                       tile.IHC_BottomLeft[::-1])).reshape((-1, 1, 2))],
                      isClosed=True,
                      color=(255, 0, 0),
                      thickness=thickness)



    cv2.imwrite('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_e_Tiles.jpg', img_HE)


    img_IHC = img_IHC[13000:, :, :]
    cv2.imwrite('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_l_Tiles.jpg', img_IHC)



def compute_feature_points_new_location(ratio, original_location):
    original_location = np.array(original_location)
    new_location = np.zeros_like(original_location)
    new_location[:, 0] = original_location[:, 0] * ratio['H']
    new_location[:, 1] = original_location[:, 1] * ratio['W']
    return new_location.tolist()


def get_transforms(points_HE, points_IHC, triangles) -> list:
	Transform_Matrices = []
	for tri in triangles:
		p_HE = [points_HE[t] for t in tri]
		p_IHC = [points_IHC[t] for t in tri]

		Transform_Matrices.append(cv2.getAffineTransform(np.float32(p_HE), np.float32(p_IHC)))

	return Transform_Matrices


def compute_transformed_tiles(tiles, transformations):
    for tile in tiles:
        transform = transformations[tile.in_simplex]
        tile.set_IHC_properties(transform)



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


HE_slide = openslide.open_slide('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/HE/19-5229_2_1_e.mrxs')
IHC_slide = openslide.open_slide('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/19-5229_2_1_l.mrxs')

dims_10_Mag = {'HE': {'W': HE_slide.level_dimensions[2][0],
                     'H': HE_slide.level_dimensions[2][1]
                      },
              'IHC': {'W': IHC_slide.level_dimensions[2][0],
                      'H': IHC_slide.level_dimensions[2][1]
                      }
              }

dims_current = {'HE': {'W': HE_slide.level_dimensions[3][0],
                       'H': HE_slide.level_dimensions[3][1]
                       },
                'IHC': {'W': IHC_slide.level_dimensions[3][0],
                        'H': IHC_slide.level_dimensions[3][1]
                        }
                }

dims_thumb = {'HE': {'W': 875,
                     'H': 2000
                     },
              'IHC': {'W': 875,
                      'H': 2000
                      }
              }


ratio_small_to_current = {'HE': {'H': dims_current['HE']['H'] / dims_thumb['HE']['H'],
                                 'W': dims_current['HE']['W'] / dims_thumb['HE']['W']
                                 },
                          'IHC': {'H': dims_current['IHC']['H'] / dims_thumb['IHC']['H'],
                                  'W': dims_current['IHC']['W'] / dims_thumb['IHC']['W']
                                  }
                          }

ratio_current_to_10MAG = {'HE': {'H': dims_10_Mag['HE']['H'] / dims_current['HE']['H'],
                                 'W': dims_10_Mag['HE']['W'] / dims_current['HE']['W']
                                 },
                          'IHC': {'H': dims_10_Mag['IHC']['H'] / dims_current['IHC']['H'],
                                  'W': dims_10_Mag['IHC']['W'] / dims_current['IHC']['W']
                                  }
                          }


feature_points_original = {'HE': [(324, 217), (306, 324), (268, 383), (239, 404), (209, 369), (409, 500), (428, 532), (407, 587), (366, 569),
                                  (376, 552), (378, 493), (439, 409), (312, 391), (267, 445), (224, 458), (156, 387), (188, 319), (240, 219),
                                  (339, 119), (463, 81), (592, 237), (455, 139), (643, 189), (685, 271), (681, 327), (385, 296), (338, 278),
                                  (249, 273), (364, 347), (455, 351), (483, 385), (555, 667), (545, 735), (375, 688), (690, 375), (756, 365),
                                  (642, 643), (555, 526), (594, 547), (555, 638), (487, 658), (456, 606), (473, 564), (512, 565)],
                           'IHC': [(1258, 193), (1228, 305), (1180, 351), (1153, 366), (1131, 339), (1318, 485), (1333, 531), (1307, 578),
                                   (1268, 545), (1280, 518), (1284, 471), (1351, 404), (1230, 369), (1187, 409), (1132, 418), (1071, 347),
                                   (1105, 295), (1180, 195), (1264, 121), (1414, 78), (1509, 244), (1401, 138), (1559, 199), (1580, 291), (1574, 345),
                                   (1305, 282), (1259, 261), (1179, 239), (1285, 325), (1366, 348), (1389, 383), (1422, 666), (1393, 724),
                                   (1261, 668), (1600, 407), (1666, 396), (1522, 647), (1455, 531), (1479, 546), (1441, 638), (1374, 648),
                                   (1350, 603), (1372, 555), (1411, 559)]
                           }


HE_new_feature_location = compute_feature_points_new_location(ratio_small_to_current['HE'], feature_points_original['HE'])
IHC_new_feature_location = compute_feature_points_new_location(ratio_small_to_current['IHC'], feature_points_original['IHC'])

save_small_images = False
if save_small_images:
    real_sized_thumb = HE_slide.read_region((0, 0), 3, (11185, 25576)).convert('RGB')
    real_sized_thumb.save('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_e.jpg')
    real_sized_thumb = IHC_slide.read_region((0, 0), 3, (11185, 25576)).convert('RGB')
    real_sized_thumb.save('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_l.jpg')



# Centers are ordered by: (row, col)
tile_centers_HE = [(5215, 1557), (6956, 1931), (4902, 3103), (6210, 3549), (2942, 4294), (3876, 4585), (5699, 4836), (5691, 5634), (6418, 5654), (7137, 7294), (6759, 9151), (8041, 7821), (8845, 5701), (9305, 4790)]
HE_tri = Delaunay(HE_new_feature_location)
tiles = create_tiles(tile_centers_HE, ratio_current_to_10MAG, HE_tri)

transforms = get_transforms(HE_new_feature_location, IHC_new_feature_location, HE_tri.simplices)

compute_transformed_tiles(tiles, transforms)


HE_thumb = cv2.imread('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_e_TOP.jpg')
IHC_thumb = cv2.imread('/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL/Immuno_ER/thumbs/19-5229_2_1_l.jpg')
plot_tiles(HE_thumb, IHC_thumb, tiles)


print('Done')



