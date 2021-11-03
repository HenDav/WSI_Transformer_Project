import subprocess

infer = True

if infer:
    subprocess.run(['python', 'inference_Multi_REG.py',
                    '--folds', '12345',
                    #'--folds', '2',
                    #'--dataset', 'Breast',
                    '--dataset', 'TCGA',
                    #'--dataset', 'ABCTB_TCGA',
                    #'--dataset', 'CARMEL9',
                    '--num_tiles', '30',
                    '-ex', '321',
                    #'--from_epoch', '0', '16',
                    '--from_epoch', '16',
                    #'--patch_dir', r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\runs\Exp_321-ER-TestFold_2\Inference'
                    #'--save_features'
                    '--model_path', 'torchvision.models.resnet34(pretrained=True)'
                    ])
else:
    #subprocess.run(['python', 'train_reg.py',
    subprocess.run(['python', 'train_reg.py',
                    '--test_fold', '3',
                    '--epochs', '2',
                    #'--dataset', 'PORTO_PDL1',
                    #'--dataset', 'Breast',
                    #'--dataset', 'ABCTB',
                    #'--dataset', 'ABCTB_TCGA',
                    '--dataset', 'TCGA',
                    #'--target', 'ER',
                    '--target', 'ER',
                    #'--target', 'Her2',
                    #'--target', 'PDL1',
                    #'--transform_type', 'bnfrs',
                    #'--transform_type', 'hedcfrs',
                    '--batch_size', '2',
                    '--n_patches_test', '10',
                    '--n_patches_train', '100',
                    '--model', 'PreActResNets.PreActResNet50_Ron()',
                    #'--model', 'nets.resnet50_with_3FC()',
                    '--bootstrap',
                    #'--transform_type', 'aug_receptornet',
                    #'--transform_type', 'rvf',
                    '--transform_type', 'pcbnfrsc',
                    '--mag', '160',
                    '--eval_rate', '10',
                    #'-d',
                    #'-im'
                    #'--loan'
                    #'--er_eq_pr'
                    #'-time',
                    #'-baldat'
                    #'--slide_per_block'
                ])

#train_reg.py --test_fold 1 --epochs 2 --dataset LUNG --target PDL1 --batch_size 5 --n_patches_test 10 --n_patches_train 10 --model resnet50_3FC --transform_type aug_receptornet