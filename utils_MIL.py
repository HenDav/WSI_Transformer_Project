import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_RegModel_Features_location_dict(train_DataSet: str, target: str, test_fold: int):
    All_Data_Dict = {'linux': {'CAT': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_355-ER-TestFold_1',
                                                         'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features',
                                                         'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         },
                                                  'Her2': {'DataSet Name': r'FEATURES: Exp_392-Her2-TestFold_1',
                                                           'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/train_w_features',
                                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/test_w_features',
                                                           'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           },
                                                  'PR': {'DataSet Name': r'FEATURES: Exp_10-PR-TestFold_1',
                                                         'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/train_w_features',
                                                         'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/test_w_features',
                                                         'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         }
                                                  },
                                       'Fold 2': {'ER': {'DataSet Name': r'FEATURES: Exp_393-ER-TestFold_2',
                                                         'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/train_w_features',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/test_w_features',
                                                         # 'TestSet Location': r'/home/womer/project/All Data/Ran_Features/393/Test',  # This is the scrambled test set
                                                         'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         },
                                                  'PR': {'DataSet Name': r'FEATURES: Exp_20063-PR-TestFold_2',
                                                         'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/train_w_features',
                                                         'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/test_w_features',
                                                         'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         },
                                                  'Her2': {'DataSet Name': r'FEATURES: Exp_412-Her2-TestFold_2',
                                                           'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/train_w_features',
                                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/test_w_features',
                                                           'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           }
                                                  },
                                       'Fold 3': {'ER': {'DataSet Name': r'FEATURES: Exp_472-ER-TestFold_3',
                                                         'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/train_w_features',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/test_w_features',
                                                         'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         },
                                                  'Her2': {'DataSet Name': r'FEATURES: Exp_20114-Her2-TestFold_3',
                                                           'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/train_w_features',
                                                           'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/test_w_features',
                                                           'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           },
                                                  'PR': {'DataSet Name': r'FEATURES: Exp_497-PR-TestFold_3',
                                                         'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/train_w_features',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/test_w_features',
                                                         'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         }
                                                  },
                                       'Fold 4': {'ER': {'DataSet Name': r'FEATURES: Exp_542-ER-TestFold_4',
                                                         'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/train_w_features',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/test_w_features',
                                                         'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         },
                                                  'Her2': {'DataSet Name': r'FEATURES: Exp_20201-Her2-TestFold_4',
                                                           'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/train_w_features',
                                                           'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/test_w_features',
                                                           'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           },
                                                  'PR': {'DataSet Name': r'FEATURES: Exp_20207-PR-TestFold_4',
                                                         'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/train_w_features',
                                                         'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/test_w_features',
                                                         'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                         }
                                                  },
                                       'Fold 5': {'Her2': {'DataSet Name': r'FEATURES: Exp_20228-Her2-TestFold_5',
                                                           'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/train_w_features',
                                                           'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/test_w_features',
                                                           'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           }
                                                  }
                                       },
                               'CAT with Location': {
                                   'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_355-ER-TestFold_1 With Locations',
                                                     'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features_locs',
                                                     'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features_locs',
                                                     'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                     },
                                              'Her2': {
                                                  'DataSet Name': r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
                                                  'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/train_w_features',
                                                  'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/test_w_locs_w_features',
                                                  'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                              },
                                              'is_Tumor_for_Her2': {
                                                  'DataSet Name': r'FEATURES: Exp_627-is_Tumor_for_Her2-TestFold_1 With Locations',
                                                  'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_Her2_train_fold2345_patches_inference_w_features',
                                                  'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_Her2_fold1_patches_inference_w_features',
                                                  'REG Model Location': None
                                              },
                                              'PR': {'DataSet Name': r'FEATURES: Exp_10-PR-TestFold_1  With Locations',
                                                     'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/train_w_features_new',
                                                     'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/test_w_locs_w_features',
                                                     'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                     },
                                              'is_Tumor_for_PR': {
                                                  'DataSet Name': r'FEATURES: Exp_627-is_Tumor_for_PR-TestFold_1 With Locations',
                                                  'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_PR_train_fold2345_patches_inference_w_features',
                                                  'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_PR_fold1_patches_inference_w_features',
                                                  'REG Model Location': None
                                              },
                                              'ER_for_is_Tumor': {
                                                  'DataSet Name': r'FEATURES: Exp_355-ER-TestFold_1 With Locations for is_Tumor',
                                                  'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features_w_locs',
                                                  'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features_locs',
                                                  'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                              },
                                              'is_Tumor_for_ER': {
                                                  'DataSet Name': r'FEATURES: Exp_627-is_Tumor_for_ER-TestFold_1 With Locations',
                                                  'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_ER_train_fold2345_patches_inference_w_features',
                                                  'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_627-is_cancer-TestFold_1/Inference/CAT_ER_test_fold1_patches_inference_w_features',
                                                  'REG Model Location': None
                                              }
                                              },

                                   'Fold 2': {'ER_for_is_Tumor': {
                                       'DataSet Name': r'FEATURES: Exp_393-ER-TestFold_2 With Locations',
                                       'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/train_w_features_new',
                                       'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/test_w_features',
                                       'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                   },
                                       'Her2': {
                                           'DataSet Name': r'FEATURES: Exp_412-Her2-TestFold_2 With Locations',
                                           'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/train_w_features',
                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/test_w_features',
                                           'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                       },
                                       'PR': {
                                           'DataSet Name': r'FEATURES: Exp_20063-PR-TestFold_2 With Locations',
                                           'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/train_w_features_new',
                                           'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/test_w_features',
                                           'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                       },
                                       'is_Tumor_for_Her2': {
                                           'DataSet Name': r'FEATURES: Exp_641-is_Tumor_for_Her2-TestFold_2 With Locations',
                                           'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_Her2_train_fold1345_patches_inference_w_features',
                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_Her2_test_fold2_patches_inference_w_features',
                                           'REG Model Location': None
                                       },
                                       'is_Tumor_for_ER': {
                                           'DataSet Name': r'FEATURES: Exp_641-is_Tumor_for_ER-TestFold_2 With Locations',
                                           'TrainSet Location': r'features:/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_ER_train_fold1345_patches_inference_w_features',
                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_ER_test_fold2_patches_inference_w_features',
                                           'REG Model Location': None
                                       },
                                       'is_Tumor_for_PR': {
                                           'DataSet Name': r'FEATURES: Exp_641-is_Tumor_for_PR-TestFold_2 With Locations',
                                           'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_PR_train_fold1345_patches_inference_w_features',
                                           'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_641-is_cancer-TestFold_2/Inference/CAT_PR_test_fold2_patches_inference_w_features',
                                           'REG Model Location': None
                                       }
                                   }
                               },
                               'CARMEL': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_358-ER-TestFold_1',
                                                            'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/train_w_features',
                                                            'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/test_w_features',
                                                            'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                            },
                                                     'Ki67': {'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1',
                                                              'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/train_w_features',
                                                              'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/test_w_features',
                                                              'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                              },
                                                     'Her2': {'DataSet Name': None,
                                                              'TrainSet Location': None,
                                                              'TestSet Location': None,
                                                              'REG Model Location': None
                                                              },
                                                     'PR': {'DataSet Name': None,
                                                            'TrainSet Location': None,
                                                            'TestSet Location': None,
                                                            'REG Model Location': None
                                                            }
                                                     },
                                          'Fold 2': {'Ki67': {'DataSet Name': r'FEATURES: Exp_490-Ki67-TestFold_2',
                                                              'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/train_w_features',
                                                              'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/test_w_features',
                                                              'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                              }
                                                     }
                                          },
                               'CARMEL 9-11': {'Fold 1': {'ER': {
                                   'DataSet Name': r'FEATURES: Model From Exp_355-ER-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                   'TestSet Location': {'Carmel 9': None,
                                                        'Carmel 10': None,
                                                        'Carmel 11': None
                                                        }
                               },
                                                          'Ki67': {
                                                              'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                                              'TestSet Location': {
                                                                  'Carmel 9': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL9/',
                                                                  'Carmel 10': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL10/',
                                                                  'Carmel 11': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL11/'
                                                              }
                                                          },
                                                          'Her2': {'DataSet Name': None,
                                                                   'TestSet Location': {'Carmel 9': None,
                                                                                        'Carmel 10': None,
                                                                                        'Carmel 11': None
                                                                                        }
                                                                   },
                                                          'PR': {'DataSet Name': None,
                                                                 'TestSet Location': {'Carmel 9': None,
                                                                                      'Carmel 10': None,
                                                                                      'Carmel 11': None
                                                                                      }
                                                                 }
                                                          },
                                               'Fold 2': {'ER': {
                                                   'DataSet Name': r'FEATURES: Model From Exp_393-ER-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                                                   'TestSet Location': {'Carmel 9': None,
                                                                        'Carmel 10': None,
                                                                        'Carmel 11': None
                                                                        }
                                               },
                                                          'Her2': {'DataSet Name': None,
                                                                   'TestSet Location': {'Carmel 9': None,
                                                                                        'Carmel 10': None,
                                                                                        'Carmel 11': None
                                                                                        }
                                                                   },
                                                          'PR': {'DataSet Name': None,
                                                                 'TestSet Location': {'Carmel 9': None,
                                                                                      'Carmel 10': None,
                                                                                      'Carmel 11': None
                                                                                      }
                                                                 }
                                                          },
                                               'Fold 4': {'ER': {
                                                   'DataSet Name': r'FEATURES: Model From Exp_542-ER-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                                                   'TestSet Location': {
                                                       'Carmel 9': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL9',
                                                       'Carmel 10': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL10',
                                                       'Carmel 11': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL11'
                                                   }
                                               },
                                                          'Her2': {
                                                              'DataSet Name': r'FEATURES: Model From Exp_20201-Her2-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                                                              'TestSet Location': {
                                                                  'Carmel 9': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL9',
                                                                  'Carmel 10': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL10',
                                                                  'Carmel 11': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL11'
                                                              }
                                                          }
                                                          },
                                               'Fold 5': {'Her2': {
                                                   'DataSet Name': r'FEATURES: Model From Exp_20228-Her2-TestFold_5, CARMEL ONLY Slides Batch 9-11',
                                                   'TestSet Location': {
                                                       'Carmel 9': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL9',
                                                       'Carmel 10': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL10',
                                                       'Carmel 11': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL11'
                                                   }
                                               }
                                                          }
                                               },
                               'HAEMEK': {'Fold 1': {'ER': {
                                   'DataSet Name': r'FEATURES: Model From Exp_355-ER-TestFold_1, HAEMEK ONLY Slides',
                                   'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/HAEMEK'
                               },
                                                     'Ki67': {
                                                         'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1, HAEMEK ONLY Slides',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/HAEMEK'
                                                     },
                                                     'Her2': {
                                                         'DataSet Name': r'FEATURES: Model From Exp_392-Her2-TestFold_1, HAEMEK ONLY Slides',
                                                         'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/HAEMEK'
                                                     },
                                                     'PR': {
                                                         'DataSet Name': r'FEATURES: Model From Exp_20010-PR-TestFold_1, HAEMEK ONLY Slides',
                                                         'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/HAEMEK'
                                                     }
                                                     }
                                          },
                               'ABCTB': {
                                   'Fold 1': {'survival': {'DataSet Name': r'FEATURES: Exp_20094-survival-TestFold_1',
                                                           'TrainSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/train_w_features/',
                                                           'TestSet Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/test_w_features/',
                                                           'REG Model Location': r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                           }
                                              }
                               },
                               'TCGA_ABCTB->CARMEL': {
                                   'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_293-ER-TestFold_1',
                                                     'TestSet Location': {
                                                         'CARMEL': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/CARMEL1-8',
                                                         'CARMEL 9-11': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/CARMEL9-11'
                                                     }
                                                     },
                                              'Her2': {'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                                                       'TestSet Location': {
                                                           'CARMEL': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/CARMEL1-8',
                                                           'CARMEL 9-11': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/CARMEL9-11'
                                                       }
                                                       },
                                              'PR': {'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                                                     'TestSet Location': {
                                                         'CARMEL': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/CARMEL1-8',
                                                         'CARMEL 9-11': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/CARMEL9-11'
                                                     }
                                                     }
                                              }
                               },
                               'TCGA_ABCTB': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_293-ER-TestFold_1',
                                                                'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/train_inference_w_features',
                                                                'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/test_inference_w_features',
                                                                'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                                },
                                                         'Her2': {'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                                                                  'TrainSet Location': r'/home/womer/project/All Data/Ran_Features/Her2/Fold_1/Train',
                                                                  'TestSet Location': r'/home/womer/project/All Data/Ran_Features/Her2/Fold_1/Test',
                                                                  'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                                  },
                                                         'PR': {'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                                                                'TrainSet Location': r'/home/womer/project/All Data/Ran_Features/PR/Fold_1/Train',
                                                                'TestSet Location': r'/home/womer/project/All Data/Ran_Features/PR/Fold_1/Test',
                                                                'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                                }
                                                         },
                                              'Fold 2': {'ER': {'DataSet Name': r'FEATURES: Exp_299-ER-TestFold_2',
                                                                'TrainSet Location': r'/home/womer/project/All Data/Ran_Features/299/Train',
                                                                'TestSet Location': r'/home/womer/project/All Data/Ran_Features/299/Test',
                                                                'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                                },
                                                         'Her2': {'DataSet Name': None,
                                                                  'TrainSet Location': None,
                                                                  'TestSet Location': None,
                                                                  'REG Model Location': None
                                                                  },
                                                         'PR': {'DataSet Name': None,
                                                                'TrainSet Location': None,
                                                                'TestSet Location': None,
                                                                'REG Model Location': None
                                                                }
                                                         }
                                              },
                               'CARMEL_40': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_381-ER-TestFold_1',
                                                               'TrainSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/train_w_features',
                                                               'TestSet Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/test_w_features',
                                                               'REG Model Location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1200.pt'
                                                               },
                                                        'Her2': {'DataSet Name': None,
                                                                 'TrainSet Location': None,
                                                                 'TestSet Location': None,
                                                                 'REG Model Location': None
                                                                 },
                                                        'PR': {'DataSet Name': None,
                                                               'TrainSet Location': None,
                                                               'TestSet Location': None,
                                                               'REG Model Location': None
                                                               }
                                                        }
                                             }
                               },
                     'darwin': {'CAT': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_355-ER-TestFold_1',
                                                          'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Train',
                                                          'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test',
                                                          'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'
                                                          },
                                                   'Her2': {'DataSet Name': r'FEATURES: Exp_392-Her2-TestFold_1',
                                                            'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/HER2/Ran_Exp_392-TestFold_1/Train',
                                                            'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/HER2/Ran_Exp_392-TestFold_1/Test',
                                                            'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data_from_gipdeep/runs/Ran_models/Her2/CAT_TF_1_392/model_data_Epoch_1000.pt'
                                                            },
                                                   'PR': {'DataSet Name': r'FEATURES: Exp_10-PR-TestFold_1',
                                                          'TrainSet Location': None,
                                                          'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_10-PR-TestFold_1/Test/',
                                                          'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/PR/CAT_10-PR-TF_1/model_data_Epoch_1000.pt'
                                                          }
                                                   },
                                        'Fold 2': {'ER': {'DataSet Name': r'FEATURES: Exp_393-ER-TestFold_2',
                                                          'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_393-TestFold_2/Train',
                                                          'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_393-TestFold_2/Test',
                                                          'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_393-ER-TF_2/model_data_Epoch_1000.pt'
                                                          },
                                                   'PR': {'DataSet Name': r'FEATURES: Exp_20063-PR-TestFold_2',
                                                          'TrainSet Location': None,
                                                          'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_20063-PR-TestFold_2/Test/',
                                                          'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/PR/CAT_20063-PR-TF_2/model_data_Epoch_1000.pt'
                                                          },
                                                   'Her2': {'DataSet Name': r'FEATURES: Exp_412-Her2-TestFold_2',
                                                            'TrainSet Location': None,
                                                            'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_412-TestFold_2/Test/',
                                                            'REG Model Location': None
                                                            }
                                                   },
                                        'Fold 3': {'ER': {'DataSet Name': r'FEATURES: Exp_472-ER-TestFold_3',
                                                          'TrainSet Location': None,
                                                          'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_472-ER-TestFold_3/Test/',
                                                          'REG Model Location': None
                                                          },
                                                   }
                                        },
                                'ABCTB': {
                                    'Fold 1': {'survival': {'DataSet Name': r'FEATURES: Exp_20094-survival-TestFold_1',
                                                            'TrainSet Location': None,
                                                            'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Test/',
                                                            'REG Model Location': None
                                                            }
                                               }
                                },
                                'CAT with Location': {
                                    'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_355-ER-TestFold_1 With Locations',
                                                      'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Train_with_location',
                                                      'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test_with_location',
                                                      'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'
                                                      },
                                               'Her2': {
                                                   'DataSet Name': r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
                                                   'TrainSet Location': None,
                                                   'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_392-TestFold_1_With_Location/Test/',
                                                   'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data_from_gipdeep/runs/Ran_models/Her2/CAT_TF_1_392/model_data_Epoch_1000.pt'
                                               },
                                               'is_Tumor': {
                                                   'DataSet Name': r'FEATURES: Exp_627-is_Tumor-TestFold_1 With Locations',
                                                   'TrainSet Location': None,
                                                   'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/is_Tumor/Ran_Exp_627-TestFold_1_With_Location/Test/',
                                                   'REG Model Location': None
                                               },
                                               'PR': {'DataSet Name': r'FEATURES: Exp_10-PR-TestFold_1  With Locations',
                                                      'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR_is_Tumor/Train',
                                                      'TestSet Location': None,
                                                      'REG Model Location': None
                                                      },
                                               'is_Tumor_for_PR': {
                                                   'DataSet Name': r'FEATURES: Exp_627-is_Tumor_for_PR-TestFold_1 With Locations',
                                                   'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/is_Tumor_for_PR/Train',
                                                   'TestSet Location': None,
                                                   'REG Model Location': None
                                               },
                                               'ResNet34': {
                                                   'DataSet Name': r'FEATURES: Extraction via ResNet 34 pretraind model',
                                                   'TrainSet Location': None,
                                                   'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Resnet34_pretrained_features/',
                                                   'REG Model Location': None
                                               }
                                               }
                                },
                                'CARMEL': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_358-ER-TestFold_1',
                                                             'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Train',
                                                             'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Test',
                                                             'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_358-TF_1/model_data_Epoch_1000.pt'
                                                             },
                                                      'Ki67': {'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1',
                                                               'TrainSet Location': None,
                                                               'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_419-Ki67-TestFold_1/Test/',
                                                               'REG Model Location': None
                                                               },
                                                      'Her2': {'DataSet Name': None,
                                                               'TrainSet Location': None,
                                                               'TestSet Location': None,
                                                               'REG Model Location': None
                                                               },
                                                      'PR': {'DataSet Name': None,
                                                             'TrainSet Location': None,
                                                             'TestSet Location': None,
                                                             'REG Model Location': None
                                                             }
                                                      },
                                           'Fold 2': {'Ki67': {'DataSet Name': r'FEATURES: Exp_490-Ki67-TestFold_2',
                                                               'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_490-Ki67-TestFold_2/Train/',
                                                               'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_490-Ki67-TestFold_2/Test/',
                                                               'REG Model Location': None
                                                               }
                                                      }
                                           },
                                'CARMEL 9-11': {'Fold 1': {'ER': {
                                    'DataSet Name': r'FEATURES: Model From Exp_355-ER-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                    'TestSet Location': {
                                        'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Carmel9/',
                                        'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Carmel10/',
                                        'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Carmel11/'
                                    }
                                },
                                                           'Ki67': {
                                                               'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_419-Ki67-TestFold_1/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_419-Ki67-TestFold_1/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_419-Ki67-TestFold_1/Carmel11/'
                                                               }
                                                           },
                                                           'Her2': {
                                                               'DataSet Name': r'FEATURES: Model From Exp_392-Her2-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_392-TestFold_1/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_392-TestFold_1/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_392-TestFold_1/Carmel11/'
                                                               }
                                                           },
                                                           'PR': {
                                                               'DataSet Name': r'FEATURES: Model From Exp_10-PR-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_10-PR-TestFold_1/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_10-PR-TestFold_1/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_10-PR-TestFold_1/Carmel11/'
                                                               }
                                                           }
                                                           },
                                                'Fold 2': {'ER': {
                                                    'DataSet Name': r'FEATURES: Model From Exp_393-ER-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                                                    'TestSet Location': {
                                                        'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_393-TestFold_2/Carmel9/',
                                                        'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_393-TestFold_2/Carmel10/',
                                                        'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_393-TestFold_2/Carmel11/'
                                                    }
                                                },
                                                           'Her2': {
                                                               'DataSet Name': r'FEATURES: Model From Exp_412-Her2-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_412-TestFold_2/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_412-TestFold_2/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Ran_Exp_412-TestFold_2/Carmel11/'
                                                               }
                                                           },
                                                           'PR': {
                                                               'DataSet Name': r'FEATURES: Model From Exp_20063-PR-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_20063-PR-TestFold_2/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_20063-PR-TestFold_2/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Ran_Exp_20063-PR-TestFold_2/Carmel11/'
                                                               }
                                                           },
                                                           'Ki67': {
                                                               'DataSet Name': r'FEATURES: Model From Exp_490-Ki67-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                                                               'TestSet Location': {
                                                                   'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_490-Ki67-TestFold_2/Carmel9/',
                                                                   'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_490-Ki67-TestFold_2/Carmel10/',
                                                                   'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Ki67/Ran_Exp_490-Ki67-TestFold_2/Carmel11/'
                                                               }
                                                           }
                                                           },
                                                'Fold 3': {'ER': {
                                                    'DataSet Name': r'FEATURES: Model From Exp_472-ER-TestFold_3, CARMEL ONLY Slides Batch 9-11',
                                                    'TestSet Location': {
                                                        'Carmel 9': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_472-ER-TestFold_3/Carmel9/',
                                                        'Carmel 10': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_472-ER-TestFold_3/Carmel10/',
                                                        'Carmel 11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_472-ER-TestFold_3/Carmel11/'
                                                    }
                                                }
                                                           }
                                                },
                                'HAEMEK': {'Fold 1': {'ER': {
                                    'DataSet Name': r'FEATURES: Model From Exp_355-ER-TestFold_1, HAEMEK ONLY Slides',
                                    'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/HAEMEK/'
                                },
                                                      'Ki67': {
                                                          'DataSet Name': r'FEATURES: Exp_419-Ki67-TestFold_1, HAEMEK ONLY Slides',
                                                          'TestSet Location': None
                                                      },
                                                      'Her2': {
                                                          'DataSet Name': r'FEATURES: Model From Exp_392-Her2-TestFold_1, HAEMEK ONLY Slides',
                                                          'TestSet Location': None
                                                      },
                                                      'PR': {
                                                          'DataSet Name': r'FEATURES: Model From Exp_20010-PR-TestFold_1, HAEMEK ONLY Slides',
                                                          'TestSet Location': None
                                                      }
                                                      }
                                           },
                                'TCGA_ABCTB->CARMEL': {
                                    'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_293-ER-TestFold_1',
                                                      'TestSet Location': {
                                                          'CARMEL': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/ABCTB_TCGA_>CARMEL/CARMEL1-8',
                                                          'CARMEL 9-11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/ABCTB_TCGA_>CARMEL/CARMEL9-11'
                                                      }
                                                      },
                                               'Her2': {'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                                                        'TestSet Location': {'CARMEL': None,
                                                                             'CARMEL 9-11': None
                                                                             }
                                                        },
                                               'PR': {'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                                                      'TestSet Location': {'CARMEL': None,
                                                                           'CARMEL 9-11': None
                                                                           }
                                                      }
                                               }
                                },
                                'TCGA_ABCTB': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_293-ER-TestFold_1',
                                                                 'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Train',
                                                                 'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test',
                                                                 'REG Model Location': None
                                                                 },
                                                          'Her2': {'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                                                                   'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Fold_1/Train',
                                                                   'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Fold_1/Test',
                                                                   'REG Model Location': None
                                                                   },
                                                          'PR': {'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                                                                 'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Fold_1/Train',
                                                                 'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Fold_1/Test',
                                                                 'REG Model Location': None
                                                                 }
                                                          },
                                               'Fold 2': {'ER': {'DataSet Name': r'FEATURES: Exp_299-ER-TestFold_2',
                                                                 'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Train',
                                                                 'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Test',
                                                                 'REG Model Location': None
                                                                 },
                                                          'Her2': {'DataSet Name': None,
                                                                   'TrainSet Location': None,
                                                                   'TestSet Location': None,
                                                                   'REG Model Location': None
                                                                   },
                                                          'PR': {'DataSet Name': None,
                                                                 'TrainSet Location': None,
                                                                 'TestSet Location': None,
                                                                 'REG Model Location': None
                                                                 }
                                                          }
                                               },
                                'CARMEL_40': {'Fold 1': {'ER': {'DataSet Name': r'FEATURES: Exp_381-ER-TestFold_1',
                                                                'TrainSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Train',
                                                                'TestSet Location': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Test',
                                                                'REG Model Location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_381-TF_1/model_data_Epoch_1200.pt'
                                                                },
                                                         'Her2': {'DataSet Name': None,
                                                                  'TrainSet Location': None,
                                                                  'TestSet Location': None,
                                                                  'REG Model Location': None
                                                                  },
                                                         'PR': {'DataSet Name': None,
                                                                'TrainSet Location': None,
                                                                'TestSet Location': None,
                                                                'REG Model Location': None
                                                                }
                                                         }
                                              }
                                }
                     }

    if '+is_Tumor' in target:
        receptor = target.split('+')[0]
        if receptor == 'Her2':
            return All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)][receptor], \
                   All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)]['is_Tumor_for_Her2']

        elif receptor == 'PR':
            return All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)][receptor], \
                   All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)]['is_Tumor_for_PR']

        elif receptor == 'ER':
            return All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)]['ER_for_is_Tumor'], \
                   All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)]['is_Tumor_for_ER']

    else:
        return All_Data_Dict[sys.platform][train_DataSet]['Fold ' + str(test_fold)][target]


def dataset_properties_to_location(dataset_name_list: list, receptor: str, test_fold: int, is_train: bool = False):
    # Basic data definition:
    if sys.platform == 'darwin':
        dataset_full_data_dict = {'TCGA_ABCTB':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Train',
                                                    'Test': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test',
                                                    'Dataset name': r'FEATURES: Exp_293-ER-TestFold_1'
                                                }
                                            }
                                       },
                                  'CAT':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Train',
                                                    'Test': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test',
                                                    'Dataset name': r'FEATURES: Exp_355-ER-TestFold_1',
                                                    'Regular model location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'}
                                            }
                                       },
                                  'CARMEL':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Train',
                                                    'Test': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Test',
                                                    'Dataset name': r'FEATURES: Exp_358-ER-TestFold_1',
                                                    'Regular model location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_358-TF_1/model_data_Epoch_1000.pt'
                                                }
                                            }
                                       },
                                  'CARMEL_40':
                                      {'ER':
                                          {1:
                                              {
                                                  'Train': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Train',
                                                  'Test': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Test',
                                                  'Dataset name': r'FEATURES: Exp_381-ER-TestFold_1',
                                                  'Regular model location': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_381-TF_1/model_data_Epoch_1200.pt'
                                              }
                                          }
                                      }
                                  }
    elif sys.platform == 'linux':
        dataset_full_data_dict = {'TCGA_ABCTB':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/train_inference_w_features',
                                                    'Test': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/test_inference_w_features',
                                                    'Dataset name': r'FEATURES: Exp_293-ER-TestFold_1'
                                                }
                                            }
                                       },
                                  'CAT':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features',
                                                    'Test': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features',
                                                    'Dataset name': r'FEATURES: Exp_355-ER-TestFold_1',
                                                    'Regular model location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'}
                                            }
                                       },
                                  'CARMEL':
                                      {'ER':
                                           {1:
                                                {
                                                    'Train': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/train_w_features',
                                                    'Test': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/test_w_features',
                                                    'Dataset name': r'FEATURES: Exp_358-ER-TestFold_1',
                                                    'Regular model location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                                                }
                                            }
                                       },
                                  'CARMEL_40':
                                      {'ER':
                                          {1:
                                              {
                                                  'Train': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/train_w_features',
                                                  'Test': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/test_w_features',
                                                  'Dataset name': r'FEATURES: Exp_358-ER-TestFold_1',
                                                  'Regular model location': r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1200.pt'
                                              }
                                          }
                                      }
                                  }

    dataset_location_list = []

    if receptor == 'ER_Features':
        receptor = 'ER'
    for dataset in dataset_name_list:
        location = dataset_full_data_dict[dataset][receptor][test_fold]['Train' if is_train else 'Test']
        dataset_name = dataset_full_data_dict[dataset][receptor][test_fold]['Dataset name']
        regular_model_location = dataset_full_data_dict[dataset][receptor][test_fold]['Regular model location']
        dataset_location_list.append([dataset, location, dataset_name, regular_model_location])

    return dataset_location_list


def save_all_slides_and_models_data(all_slides_tile_scores, all_slides_final_scores,
                                    all_slides_weights_before_softmax, all_slides_weights_after_softmax,
                                    models, Output_Dirs, Epochs, data_path, true_test_path: str = ''):
    # Save slide scores to file:
    for num_model in range(len(models)):
        if type(Output_Dirs) == str:
            output_dir = Output_Dirs
        elif type(Output_Dirs) is list:
            output_dir = Output_Dirs[num_model]

        epoch = Epochs[num_model]
        model = models[num_model]

        full_output_dir = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores', 'Epoch_' + str(epoch),
                                       true_test_path)

        if not os.path.isdir(full_output_dir):
            Path(full_output_dir).mkdir(parents=True)

        model_bias_filename = 'bias.xlsx'
        full_model_bias_filename = os.path.join(full_output_dir, model_bias_filename)
        if not os.path.isfile(full_model_bias_filename):
            try:  # In case this part in not packed in Sequential we'll need this try statement
                last_layer_bias = model.classifier[0].bias.detach().cpu().numpy()
            except TypeError:
                last_layer_bias = model.classifier.bias.detach().cpu().numpy()

            last_layer_bias_diff = last_layer_bias[1] - last_layer_bias[0]

            last_layer_bias_DF = pd.DataFrame([last_layer_bias_diff])
            last_layer_bias_DF.to_excel(full_model_bias_filename)

        if type(all_slides_tile_scores) == dict:
            all_slides_tile_scores_REG = all_slides_tile_scores['REG']
            all_slides_final_scores_REG = all_slides_final_scores['REG']
            all_slides_tile_scores = all_slides_tile_scores['MIL']
            all_slides_final_scores = all_slides_final_scores['MIL']

            all_slides_tile_scores_REG_DF = pd.DataFrame(all_slides_tile_scores_REG[num_model]).transpose()
            all_slides_final_scores_REG_DF = pd.DataFrame(all_slides_final_scores_REG[num_model]).transpose()

            tile_scores_file_name_REG = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores',
                                                     'Epoch_' + str(epoch), true_test_path, 'tile_scores_REG.xlsx')
            slide_score_file_name_REG = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores',
                                                     'Epoch_' + str(epoch), true_test_path, 'slide_scores_REG.xlsx')

            all_slides_tile_scores_REG_DF.to_excel(tile_scores_file_name_REG)
            all_slides_final_scores_REG_DF.to_excel(slide_score_file_name_REG)

        all_slides_tile_scores_DF = pd.DataFrame(all_slides_tile_scores[num_model]).transpose()
        all_slides_final_scores_DF = pd.DataFrame(all_slides_final_scores[num_model]).transpose()
        all_slides_weights_before_sofrmax_DF = pd.DataFrame(all_slides_weights_before_softmax[num_model]).transpose()
        all_slides_weights_after_softmax_DF = pd.DataFrame(all_slides_weights_after_softmax[num_model]).transpose()

        tile_scores_file_name = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores', 'Epoch_' + str(epoch),
                                             true_test_path, 'tile_scores.xlsx')
        slide_score_file_name = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores', 'Epoch_' + str(epoch),
                                             true_test_path, 'slide_scores.xlsx')
        tile_weights_before_softmax_file_name = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores',
                                                             'Epoch_' + str(epoch), true_test_path,
                                                             'tile_weights_before_softmax.xlsx')
        tile_weights_after_softmax_file_name = os.path.join(data_path, output_dir, 'Inference', 'Tile_Scores',
                                                            'Epoch_' + str(epoch), true_test_path,
                                                            'tile_weights_after_softmax.xlsx')

        all_slides_tile_scores_DF.to_excel(tile_scores_file_name)
        all_slides_final_scores_DF.to_excel(slide_score_file_name)
        all_slides_weights_before_sofrmax_DF.to_excel(tile_weights_before_softmax_file_name)
        all_slides_weights_after_softmax_DF.to_excel(tile_weights_after_softmax_file_name)

        logging.info('Tile scores for model {}/{} has been saved !'.format(num_model + 1, len(models)))


def extract_tile_scores_for_slide(all_features, models):
    """
    If all_features has shape[0] == 1024, than it;s originated from train type Receptor + is_Tumor.
    In that case we'll need only the first 512 features to compute the tile scores.
    """
    if all_features.shape[0] == 1024:
        all_features = all_features[:512, :]

    # Save tile scores and last models layer bias difference to file:
    tile_scores_list = []
    for index in range(len(models)):
        model = models[index]
        # Compute for each tile the multiplication between its feature vector and the last layer weight difference
        # vector:
        try:  # In case this part in not packed in Sequential we'll need this try statement
            last_layer_weights = model.classifier[0].weight.detach().cpu().numpy()
        except TypeError:
            last_layer_weights = model.classifier.weight.detach().cpu().numpy()

        f = last_layer_weights[1] - last_layer_weights[0]
        mult = np.matmul(f, all_features.detach().cpu())

        if len(mult.shape) == 1:
            tile_scores_list.append(mult)
        else:
            tile_scores_list.append(mult[:, index])

    return tile_scores_list