import deepfinder.utils.objl as ol
import os, warnings

class Dataloader:
    def __init__(self):
        self.path_data = []
        self.path_target = []
        self.objl_train = []
        self.objl_valid = []
        self.tomo_idx = 0

    def __call__(self, path_dset):
        path_train = os.path.join(path_dset, 'train')
        path_valid = os.path.join(path_dset, 'valid')

        if os.path.isdir(path_train):
            self.load_content(path_train)
        else:
            raise Exception('DeepFinder: train folder has not been found.')

        if os.path.isdir(path_valid):
            self.load_content(path_valid)
        else:
            #raise Warning('DeepFinder: valid folder has not been found.')
            print('DeepFinder: valid folder has not been found.')

        return self.path_data, self.path_target, self.objl_train, self.objl_valid

    def load_content(self, path):
        for fname in os.listdir(path):
            # if fname does not start with '.' (invisible temporary files) and end with '_objl.xml'
            if fname[0] is not '.' and fname.endswith('_objl.xml'):
                fprefix = fname[:-9]  # remove '_objl.xml'
                fname_tomo = fprefix + '.mrc'
                fname_target = fprefix + '_target.mrc'

                fname = os.path.join(path, fname)
                fname_tomo = os.path.join(path, fname_tomo)
                fname_target = os.path.join(path, fname_target)

                self.path_data.append(fname_tomo)
                self.path_target.append(fname_target)

                objl = ol.read(fname)
                for obj in objl:  # attribute tomo_idx to objl
                    obj['tomo_idx'] = self.tomo_idx

                if path.endswith('train'):
                    self.objl_train += objl
                elif path.endswith('valid'):
                    self.objl_valid += objl

                self.tomo_idx += 1

    def load_copick_datasets(self, copickPath, train_instance, tomoIDs = None):
    
        data_list   = {}; target_list = {}

        copickRoot = CopickRootFSSpec.from_file(copickPath)
        if tomoIDs is None:  tomoIDs = [run.name for run in copickRoot.runs]

        print(f'Loading Targets and Tomograms for the Following Runs: {list(tomoIDs)}') 
        for idx in tqdm(range(len(tomoIDs))):
            target_list[tomoIDs[idx]] = copicktools.get_copick_segmentation( copickRoot.get_run(tomoIDs[idx]), train_instance.labelName, train_instance.labelUserID)[:] 
            data_list[tomoIDs[idx]] = copicktools.read_copick_tomogram_group(copickRoot, train_instance.voxelSize, train_instance.tomoAlg, tomoIDs[idx])[0][:]

            if data_list[tomoIDs[idx]].shape != target_list[tomoIDs[idx]].shape:
                print(f'DeepFinder Message: tomogram and target for run {tomoIDs[idx]} are not of same size!')
                sys.exit()

        return data_list, target_list                


#path_dset = '/net/serpico-fs2/emoebel/cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/'
#path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)