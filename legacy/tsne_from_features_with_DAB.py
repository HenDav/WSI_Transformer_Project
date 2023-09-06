import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.manifold import TSNE
import os
import pickle
from heatmap_utils import get_combined_heatmap
import copy
from Dataset_Maker.dataset_utils import get_datasets_dir_dict
import openslide


def rgb2dub(im_RGB):
    r, c, _ = im_RGB.shape
    im_RGB = (im_RGB * 255 + 1) / 256 # just to keep values above zero

    # set hematoxylin, eosin, and DAB colors

    # B = np.array([255, 255, 255]) / 255 # stain backgroud color.

    border_vals = []
    for i in range(2):
        for j in range(2):
            val = im_RGB[i,j]
            border_vals.append(val)

    border = np.percentile(np.array(border_vals), 50, axis=0)
    if border.min() < 0.95:
        border = [0.98, 0.98, 0.98]
    else:
        border = [i for i in border]

    B = np.array(border) # stain backgroud color.
    M = np.array([[0.341, 0.9647, 0.569],
              [0.365, 0.7569, 0.467],
              [0.655, 0.9725, 0.463]])

    # working in optical density(OD) space
    B_3d = B
    im_OD = -np.log10(im_RGB / B_3d)
    M = -np.log10(M)

    # transformation matrices
    norm_params = np.sqrt((M ** 2).sum(axis=0))
    M_norm = M / np.stack([norm_params,norm_params,norm_params])
    T_HEDtoRGB = M_norm # transformation matrix from HED intensities to OD of RGB
    T_RGBtoHED = np.linalg.inv(M_norm) # transformation matrix from OD of the RGB image to intensities of HED

    # im_OD_flat = reshape(im_OD, [], 3);
    im_OD_flat = np.reshape(im_OD, [-1, 3])
    I_HED_flat = (T_RGBtoHED @ im_OD_flat.T).T # transform input image to HEB intensities
    I_HED = np.reshape(I_HED_flat, [r, c, 3])

    # decompose to H E D components
    I_H = np.zeros((r, c, 3)) # intensity of H
    I_H[:,:, 0] = I_HED[:,:, 0]
    I_H_flat = np.reshape(I_H, [-1,3])

    I_E = np.zeros((r, c, 3)) # intensity of H
    I_E[:,:, 1] = I_HED[:,:, 1]
    I_E_flat = np.reshape(I_E, [-1,3])

    I_D = np.zeros((r, c, 3))
    I_D[:,:, 2] = I_HED[:,:, 2]
    I_D_flat = np.reshape(I_D, [-1,3])

    # trasnform back each component to RGB
    im_RGB_H_flat = 10. ** (-T_HEDtoRGB @ I_H_flat.T).T * B.T
    im_RGB_H = np.reshape(im_RGB_H_flat, [r, c, 3])

    im_RGB_E_flat = 10. ** (-T_HEDtoRGB @ I_E_flat.T).T * B.T
    im_RGB_E = np.reshape(im_RGB_E_flat, [r, c, 3])

    im_RGB_D_flat = 10. ** (-T_HEDtoRGB @ I_D_flat.T).T * B.T
    im_RGB_D = np.reshape(im_RGB_D_flat, [r, c, 3])

    return np.clip(im_RGB_H, 0 , 1), np.clip(im_RGB_E, 0 , 1), np.clip(im_RGB_D, 0 , 1), np.clip(I_HED, 0, 1)


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def load_im(slide_path, im_size, margin_factor, x, y):
    slide = openslide.open_slide(slide_path)
    ihc_im = cv2.cvtColor(np.array(slide.read_region((x, y), 2, (256, 256)).convert('RGB').resize((im_size, im_size))), cv2.COLOR_RGB2BGR)
    #center_maring = int(ihc_im.shape[0] * margin_factor)
    #ihc_im = ihc_im[center_maring:-center_maring, center_maring:-center_maring]
    #ihc_im = cv2.resize(ihc_im, dsize=(im_size, im_size), interpolation=cv2.INTER_AREA)
    return ihc_im

def generate_df(folders):
    dataset_dict = get_datasets_dir_dict('Carmel 9-11')
    j=0
    slides_mat = None
    for folder in folders:
        dataset_path = dataset_dict[folder.split('/')[-1]]
        files = os.listdir(folder)

        for file in files:
            path = os.path.join(folder,file)

            with open(path, 'rb') as filehandle:
                inference_data = pickle.load(filehandle)
            if len(inference_data) == 8:
                labels, targets, scores, patch_scores, slide_names, features, batch_number, tile_location = inference_data
            else:
                continue

            for i, name in enumerate(slide_names):
                first_nan = np.where(np.isnan(tile_location[i]))[0]
                if len(first_nan) > 0:
                    first_nan = first_nan[0]
                    if first_nan == 0:
                        continue
                    length = first_nan -1
                else:
                    length = 500
                num_patches_from_slide = 50
                to_sample = max(min(length, num_patches_from_slide), 1)
                indices = np.arange(length)
                np.random.shuffle(indices)
                features_sampled = features[i,0, indices[:to_sample]]
                patch_scores_sampled = patch_scores[i, indices[:to_sample]]
                locations_sampled = tile_location[i, indices[:to_sample]]
                slide_path = np.array([os.path.join(dataset_path, name)])
                label = np.array([labels[i]])
                slide_mat = np.concatenate([np.repeat(slide_path[np.newaxis, np.newaxis, 0], len(patch_scores_sampled), 0), np.repeat(label[np.newaxis, np.newaxis, 0], len(patch_scores_sampled), 0), patch_scores_sampled[:,np.newaxis], locations_sampled, features_sampled], axis=-1)
                if slides_mat is None:
                    slides_mat = slide_mat
                else:
                    slides_mat = np.concatenate([slides_mat, slide_mat])
                j+=1
                if j % 25 == 0:
                    print(f'read {j} slides')
    slides_df = pd.DataFrame(slides_mat)
    slides_df = slides_df.rename(columns={slides_df.columns[0]: 'slide_path', slides_df.columns[1]: 'label', slides_df.columns[2]: 'score', slides_df.columns[3]: 'y', slides_df.columns[4]: 'x'})
    print(f'done read {j} slides')
    return slides_df

# root_dir directory that contains the images listed in the excel_file:
#root_dir = r"test_tsne"
out_dir = r"Her2_tsne"
os.makedirs(out_dir, exist_ok=True)
slides_df_path = out_dir + '/slides_df.csv'
if os.path.isfile(slides_df_path):
    df = pd.read_csv(slides_df_path)
else:
    folder9 = "/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/CARMEL9"
    #folder10 = '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/CARMEL10'
    folder11 = "/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/CARMEL11"
    folders = [folder9, folder11]
    df = generate_df(folders)
    df = df.sort_values(by=['score'], ascending=False, ignore_index=True)
    df.to_csv(slides_df_path)
'''
max_df = copy.deepcopy(df)
done_ids = set()
for i, row in df.iterrows():
    if int(row['id']) in done_ids:
        max_df.drop(index=i, inplace=True, axis=0)
    else:
        done_ids.add(row['id'])
df = max_df
'''

features = np.array(df.iloc[:, 5:])
scores = np.array(df['score'])
labels = np.array(df['label'])

f_embedded = TSNE(n_components=2, init='random', random_state=0).fit_transform(features)
print("done computing tsne")

f_embedded_rot = rotate(f_embedded, origin=f_embedded.mean(axis=0), degrees=90) # just for better visualisation, depends on the data
f_embedded = f_embedded_rot
f_embedded[:, 0] = f_embedded[:, 0] * -1 # rotate up <--> down

x_coords = list(f_embedded[:, 0])
y_coords = list(f_embedded[:, 1])



minx = f_embedded[:, 0].min()
maxx = f_embedded[:, 0].max()
miny = f_embedded[:, 1].min()
maxy = f_embedded[:, 1].max()
num_bins = 20
bins_x = np.linspace(minx, maxx, num_bins)
bins_y = np.linspace(miny, maxy, num_bins)
f_quant_x = np.digitize(f_embedded[:, 0], bins_x)
f_quant_y = np.digitize(f_embedded[:, 1], bins_y)

bin2id = np.zeros((num_bins + 1, num_bins + 1), dtype='uint')
used_rows = set()

## Fill lower part (high)
for i, (x, y) in enumerate(zip(f_quant_x, f_quant_y)):
    if bin2id[x, y] != 0:
        continue
    bin2id[x, y] = i

print('################')

for i, (x, y) in enumerate(zip(f_quant_x, f_quant_y)):
    if bin2id[x, y] != 0:
        continue
    else:
        bin2id[x, y] = i


# im_size = 128
im_size = 2048
TSNE_VIS_ARR_HE = 255 * np.ones((im_size * num_bins, im_size * num_bins, 3), dtype='uint8')
#TSNE_VIS_ARR_SAL = 255 * np.ones((im_size * num_bins, im_size * num_bins, 3), dtype='uint8')
#TSNE_VIS_ARR_PDL = 255 * np.ones((im_size * num_bins, im_size * num_bins, 3), dtype='uint8')
#TSNE_VIS_ARR_DUB = 255 * np.ones((im_size * num_bins, im_size * num_bins, 3), dtype='uint8')
#TSNE_VIS_ARR_H = 255 * np.ones((im_size * num_bins, im_size * num_bins, 3), dtype='uint8')



margin_factor = 0.1
tsne_metadata = pd.DataFrame(columns = ['slide name', 'slide label', 'patch score', 'x coord in slide', 'y coord in slide', 'x coord in tsne', 'y coord in tsne'])
tsne_vis_metadata = pd.DataFrame(columns = ['slide name', 'slide label', 'patch score', 'x coord in slide', 'y coord in slide', 'x coord in tsne', 'y coord in tsne'])

for row_idx, (i, j) in enumerate(f_embedded):
    x_loc = int(float(df.iloc[row_idx]['x']))
    y_loc = int(float(df.iloc[row_idx]['y']))
    slide_path = df.iloc[row_idx]['slide_path']
    slide_name = slide_path.split('/')[-1]
    label = df.iloc[row_idx]['label']
    score = df.iloc[row_idx]['score']
    x_in_tsne = i
    y_in_tsne = j
    x_in_slide = x_loc
    y_in_slide = y_loc
    data_row = [slide_name, label, score, x_in_slide, y_in_slide, x_in_tsne, y_in_tsne]
    tsne_metadata.loc[len(tsne_metadata)] = data_row
        
for i in range(num_bins):
    for j in range(num_bins):
        row_idx = bin2id[i, j]
        if row_idx == 0:
            continue

        used_rows.add(row_idx)
        x_loc = int(float(df.iloc[row_idx]['x']))
        y_loc = int(float(df.iloc[row_idx]['y']))
        slide_path = df.iloc[row_idx]['slide_path']
        slide_name = slide_path.split('/')[-1]
        label = df.iloc[row_idx]['label']
        score = df.iloc[row_idx]['score']
        x_in_tsne = i
        y_in_tsne = j
        x_in_slide = x_loc
        y_in_slide = y_loc
        data_row = [slide_name, label, score, x_in_slide, y_in_slide, x_in_tsne, y_in_tsne]
        tsne_vis_metadata.loc[len(tsne_vis_metadata)] = data_row
        
        he_im = load_im(slide_path, im_size, margin_factor=margin_factor, x=x_loc, y=y_loc)

        TSNE_VIS_ARR_HE[j * im_size:(j + 1) * im_size, i * im_size:(i + 1) * im_size] = he_im
        '''
        PDL_path = root_dir + '/PDL1(SP142)-Springbio/' + he_name.replace('HE', 'PDL1(SP142)-Springbio')
        if os.path.exists(PDL_path):
            pdl_im = load_im(PDL_path, im_size, margin_factor=margin_factor)
            TSNE_VIS_ARR_PDL[j * im_size:(j + 1) * im_size, i * im_size:(i + 1) * im_size] = pdl_im
            im_RGB_H, im_RGB_E, im_RGB_D, I_HED = rgb2dub(pdl_im[...,::-1]/255)
            TSNE_VIS_ARR_DUB[j * im_size:(j + 1) * im_size, i * im_size:(i + 1) * im_size] = (255 * im_RGB_D).astype('uint8')
            TSNE_VIS_ARR_H[j * im_size:(j + 1) * im_size, i * im_size:(i + 1) * im_size] = (255 * im_RGB_H).astype('uint8')
        


test = copy.deepcopy(bin2id) * 1.
for i in range(num_bins+1):
    for j in range(num_bins+1):
        if test[i,j] != 0:
            test[i, j] = scores[bin2id[i, j]]
'''

print("done computing visualization")



tsne_path = out_dir + '/tsne_features_scores.png'
bin_tsne_path = out_dir + '/tsne_features_labels.png'
he_path = out_dir + f'/tsne_he_{im_size}x{im_size}.jpg'
metadata_path = out_dir + '/metadata.csv'
vis_metadata_path = out_dir + '/metadata_for_image.csv'
#pdl_path = out_dir + '/tsne_pdl.png'
#dab_path = out_dir + '/tsne_pdl_dab.png'
#h_path = out_dir + '/tsne_pdl_h.png'

cv2.imwrite(he_path, TSNE_VIS_ARR_HE)
#cv2.imwrite(pdl_path, TSNE_VIS_ARR_PDL)
#cv2.imwrite(dab_path, TSNE_VIS_ARR_DUB)
#cv2.imwrite(h_path, TSNE_VIS_ARR_H)

tsne_metadata.to_csv(metadata_path)
tsne_vis_metadata.to_csv(vis_metadata_path)

plt.figure(figsize=(12, 12))

colors = scores
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(f_embedded[:, 0], -1 * f_embedded[:, 1], c=np.argsort(scores.astype(float))/(len(scores)-1), cmap=cm)
plt.colorbar(sc)
plt.xlim([-1.1 * np.abs(f_embedded).max(), 1.1 * np.abs(f_embedded).max()])
plt.ylim([-1.1 * np.abs(f_embedded).max(), 1.1 * np.abs(f_embedded).max()])
plt.title('colored by model score')
plt.axis('off')
plt.savefig(tsne_path)
plt.close()


plt.figure(figsize=(12, 12))
sc = plt.scatter(f_embedded[:, 0], -1 * f_embedded[:, 1], c=labels.astype(float), cmap=cm)
plt.colorbar(sc)
plt.xlim([-1.1 * np.abs(f_embedded).max(), 1.1 * np.abs(f_embedded).max()])
plt.ylim([-1.1 * np.abs(f_embedded).max(), 1.1 * np.abs(f_embedded).max()])
plt.title('colored by slide label')
plt.axis('off')
plt.savefig(bin_tsne_path)
plt.close()
'''
df = pd.DataFrame({'amir H&E im_name': list(df['amir H&E im_name']),
                  'original H&E im_name': list(df['original H&E im_name']),
                  'id': list(df['id']),
                   'x': x_coords,
                   'y': y_coords,
                   'score (after softmax)': list(df['score (after softmax)']),
                   'gt label': list(df['gt label'])
                   })

df.to_excel(out_dir + '/tsne_metadata.xlsx', index=False)
'''

print("done saving files")
