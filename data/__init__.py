'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        if dataset_opt['Distributed']:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_opt['batch_size'],
                num_workers=dataset_opt['num_workers'],
                sampler=DistributedSampler(dataset),
                pin_memory=True)
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=dataset_opt['use_shuffle'],
                num_workers=dataset_opt['num_workers'],
                pin_memory=True) 
    elif phase == 'val' or phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, degrade=None):
    '''create dataset'''
    # from data.LRHR_dataset import LRHRDataset as D
    from data.dataloader_pair import RestoreDataset as D
    if phase == 'train':
        if degrade == 'blur':
            dataset = D(dataroot=dataset_opt['dataroot_blur'],
                        degrade_type=degrade,
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
                        )
        elif degrade == 'rain':
            dataset = D(dataroot=dataset_opt['dataroot_rain'],
                        degrade_type=degrade,
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
            )
        elif degrade == 'noise':
            dataset = D(dataroot=dataset_opt['dataroot_noise'],
                        degrade_type=degrade,
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
                        )
        elif degrade == 'lowlight':
            dataset = D(dataroot=dataset_opt['dataroot_light'],
                        degrade_type=degrade,
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
                        )
        elif not degrade:
            dataset = D(dataroot=dataset_opt['dataroot'],
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
                        )
        else:
            raise NotImplementedError('Degrade type [{:s}] is not included'.format(degrade))
    elif phase == 'test':
        dataset = D(dataroot=dataset_opt['dataroot'],
                        degrade_type=degrade,
                        patch_size=dataset_opt['patch_size'],
                        phase=phase,
                        data_len=dataset_opt['data_len'],
            ) 
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created with {:d} samples'.format(dataset.__class__.__name__,
                                                            degrade, len(dataset)))
    print('Dataset [{:s} - {:s}] is created with {:d} samples'.format(dataset.__class__.__name__,
                                                            degrade, len(dataset)))
    return dataset
