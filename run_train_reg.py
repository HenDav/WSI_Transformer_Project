import subprocess

infer = True

if infer:
    subprocess.run(['python', 'inference_Multi_REG.py',
                    '--folds', '12345',
                    #'--folds', '2',
                    #'--dataset', 'Breast',
                    #'--dataset', 'TCGA',
                    '--dataset', 'ABCTB_TCGA',
                    #'--dataset', 'TMA',
                    #'--num_tiles', '30',
                    '--num_tiles', '5',
                    #'-ex', '321',
                    #'-ex', '561',
                    '-ex', '621',
                    #'--from_epoch', '0', '16',
                    '--from_epoch', '1000',
                    #'--from_epoch', '16',
                    #'--patch_dir', r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\runs\Exp_321-ER-TestFold_2\Inference',
                    #'--patch_dir', r'C:\Pathnet_results\MIL_general_try4\CAT_runs\Her2\exp392\Inference\test_w_locs\temp',
                    #'--save_features',
                    #'--model_path', 'torchvision.models.resnet34(pretrained=True)',
                    #'--resume', '1'
                    ])
else:
    #subprocess.run(['python', 'train_reg.py',
    subprocess.run(['python', 'train_reg.py',
                    '--test_fold', '1',
                    #'--epochs', '10',
                    #'--dataset', 'PORTO_PDL1',
                    #'--dataset', 'Breast',
                    #'--dataset', 'ABCTB',
                    '--dataset', 'ABCTB_TCGA',
                    #'--dataset', 'TMA',
                    #'--dataset', 'LEUKEMIA',
                    #'--dataset', 'TCGA',
                    #'--target', 'ER',
                    #'--target', 'temp',
                    #'--target', 'is_tel_aml_B',
                    '--target', 'ER+PR+Her2',
                    #'--target', 'Her2',
                    #'--batch_size', '1',
                    '--batch_size', '4',
                    '--n_patches_test', '1',
                    '--n_patches_train', '1',
                    #'--model', 'PreActResNets.PreActResNet50_Ron()',
                    #'--model', "StereoSphereRes(512, input_channels=3, sphereface_size=12, train_ae=0, multi_inputs_depth=0, bm=True)",
                    #'--model', 'PreActResNets.PreActResNet50_Ron(train_classifier_only=True)',
                    '--model', 'PreActResNets.PreActResNet50_Ron(num_classes=6)',
                    #'--model', 'nets.MyResNet34(train_classifier_only=True)',
                    #'--model', "nets.ResNet50()",
                    #'--bootstrap',
                    #'--transform_type', 'aug_receptornet',
                    #'--transform_type', 'rvf',
                    #'--transform_type', 'pcbnfrsc',
                    '--transform_type', 'none',
                    #'--mag', '7',
                    '--eval_rate', '10',
                    #'-tl', 'ex321,epoch16',
                    #'-d',
                    #'-im'
                    #'--loan'
                    #'--er_eq_pr'
                    #'-time',
                    #'-baldat'
                    #'--slide_per_block'
                    #'--RAM_saver'
                ])

#train_reg.py --test_fold 1 --epochs 2 --dataset LUNG --target PDL1 --batch_size 5 --n_patches_test 10 --n_patches_train 10 --model resnet50_3FC --transform_type aug_receptornet