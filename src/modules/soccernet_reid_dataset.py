import os
import glob
import re
import os.path as osp
import io
import json
from PIL import Image
from torchreid.reid.data import ImageDataset
from torchreid.reid.data import register_image_dataset
from tqdm import tqdm

class SoccerNetReIDDataset(ImageDataset):
    """
    SoccerNet re-identification dataset.

    Loads images from a local SoccerNet crops folder, prepares train/query/gallery
    splits, and optionally packs all images into a single binary file with a JSON
    index for faster subsequent reads.
    """

    dataset_dir = 'C:/Users/ciroc/Desktop/AV_project/data/datasets/ReidCrop/SoccerNet_Crops_for_REID' 

    BIN_FILE_NAME = "soccernet_packed.bin"
    IDX_FILE_NAME = "soccernet_index.json"

    def __init__(self, root='', **kwargs):
        """
        Initialize dataset.

        Args:
            - root: base path used to locate dataset_dir; if root already points to
            dataset_dir it is used directly.
            
        - Builds train, query, gallery lists by scanning subfolders.
        - Removes query identities not present in gallery.
        - Packs images into a binary cache and writes an index file if not present.
        - Loads the index into self.offset_index and calls the parent constructor.

        No return value.
        """
        self.root = osp.abspath(osp.expanduser(root))

        if not self.root.endswith(self.dataset_dir):
             self.dataset_dir_path = osp.join(self.root, self.dataset_dir)
             if not osp.exists(self.dataset_dir_path): 
                 self.dataset_dir_path = self.root
        else:
             self.dataset_dir_path = self.root

        self.train_dir = osp.join(self.dataset_dir_path, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir_path, 'query')
        self.gallery_dir = osp.join(self.dataset_dir_path, 'bounding_box_test')

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        gallery_pids = set([pid for _, pid, _ in gallery])
        valid_query = []
        skipped_queries = 0
        for img_path, pid, camid in query:
            if pid in gallery_pids:
                valid_query.append((img_path, pid, camid))
            else:
                skipped_queries += 1

        if skipped_queries > 0:
            print(f"[ DEBUG | {skipped_queries} IDs were removed frome the 'Query' dir, because they were not in the 'Gallery' ]")
        query = valid_query

        self.full_data_list = train + query + gallery

        self.bin_path = osp.join(self.dataset_dir_path, self.BIN_FILE_NAME)
        self.idx_path = osp.join(self.dataset_dir_path, self.IDX_FILE_NAME)

        if not osp.exists(self.bin_path) or not osp.exists(self.idx_path):
            print(f"[ DEBUG | Packing Binary File: {self.bin_path} ]")
            self._pack_dataset(self.full_data_list)
        else:
            print(f"[ DEBUG | Bynary cache found into '{self.bin_path}' ]")

        with open(self.idx_path, 'r') as f:
            self.offset_index = json.load(f)

        super(SoccerNetReIDDataset, self).__init__(train, query, gallery, **kwargs)

    def _pack_dataset(self, data_list):
        """
        Pack images into a single binary file and create an index.

        Args:
            - data_list: iterable of (img_path, pid, camid).
                         Writes concatenated raw image bytes to self.bin_path and a JSON mappingof img_path -> [offset, size] to self.idx_path.
        """
        index_dict = {}
        current_offset = 0

        with open(self.bin_path, 'wb') as f_bin:
            for img_path, _, _ in tqdm(data_list, desc="Packing Binary", unit="img"):
                try:
                    with open(img_path, 'rb') as f_img:
                        img_bytes = f_img.read()

                    size = len(img_bytes)
                    f_bin.write(img_bytes)

                    index_dict[img_path] = [current_offset, size]
                    current_offset += size
                except Exception as e:
                    print(f"[ EXCEPTION | Packing error: {img_path} - {e} ]")

        with open(self.idx_path, 'w') as f_idx:
            json.dump(index_dict, f_idx)

        print(f"[ DEBUG | Packing completed! Size: {current_offset / 1024 / 1024 / 1024:.2f} GB ]")

    def process_dir(self, dir_path, relabel=False):
        """
        Scan a directory for jpg images and return (img_path, pid, camid) entries.

        Args:
            - dir_path: directory to scan.
            - relabel: if True, remap original pids to consecutive labels starting at 0.
                       Uses filename pattern '<pid>_c<camid>...' to extract pid and camid.
                       Filters out pid == -1.
        
        Returns:
            a list of tuples.
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if not match: continue
            pid, _ = map(int, match.groups())
            if pid == -1: continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if not match: continue
            pid, camid = map(int, match.groups())
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data

    def __getitem__(self, index):
        """
        Return a single sample by index.

        Args:
            - index: index into self.data (provided by parent ImageDataset).
                     Reads image bytes from the packed binary file when available, otherwise
                     falls back to reading the image file. Applies self.transform if set.
        Returns: 
            a dict with keys: img, pid, camid, img_path.
        """
        item = self.data[index]

        img_path = item[0]
        pid = item[1]
        camid = item[2]

        if img_path in self.offset_index:
            offset, size = self.offset_index[img_path]
            with open(self.bin_path, 'rb') as f:
                f.seek(offset)
                img_bytes = f.read(size)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"[ EXCEPTION | Reading fallback error: {img_path} ]")
                img = Image.new('RGB', (128, 256))

        if self.transform is not None:
            img = self.transform(img)

        return {
            'img': img,
            'pid': pid,
            'camid': camid,
            'img_path': img_path 
        }

register_image_dataset('SoccerNetReIDDataset', SoccerNetReIDDataset)