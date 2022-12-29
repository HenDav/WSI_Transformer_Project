import dataset_maker
from Dataset_Maker import folds_split_per_patient

Dataset_list = ['SHEBA2', 'SHEBA3', 'SHEBA4', 'SHEBA5', 'SHEBA6']
data_dir = r'C:\ran_data\Sheba'
hospital_metadata_file = r'Sheba_Oncotype_2015-2020_09-05-22.xlsx'
fold_params = folds_split_per_patient.fold_split_params()
split_all_dataset_group = True


for Dataset in Dataset_list:
    dataset_maker.prepare_dataset_step3(Dataset=Dataset,
                                        data_dir=data_dir,
                                        fold_params=fold_params,
                                        split_all_dataset_group=split_all_dataset_group,
                                        hospital_metadata_file=hospital_metadata_file)
