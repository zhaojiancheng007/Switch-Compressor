import math
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from npy_append_array import NpyAppendArray
from torch.utils.data import Dataset

from switch_nerf.datasets.dataset_utils import get_rgb_index_mask
from switch_nerf.image_metadata import ImageMetadata
from switch_nerf.misc_utils import main_tqdm, main_print, main_log, process_log
from switch_nerf.ray_utils import get_ray_directions, get_rays, get_rays_batch
from functools import partial
import random

from dataclasses import dataclass
from einops import rearrange
from scipy import ndimage
from typing import List, Union
import tifffile


RAY_CHUNK_SIZE = 64 * 1024


class FilesystemDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, chunk_paths: List[Path], num_chunks: int,
                 scale_factor: int, disk_flush_size: int, shuffle_chunk=False):
        super(FilesystemDataset, self).__init__()
        self._device = device
        self._c2ws = torch.cat([x.c2w.unsqueeze(0) for x in metadata_items])
        self._near = near
        self._far = far
        self._ray_altitude_range = ray_altitude_range

        intrinsics = torch.cat(
            [torch.cat([torch.FloatTensor([x.W, x.H]), x.intrinsics]).unsqueeze(0) for x in metadata_items])
        if (intrinsics - intrinsics[0]).abs().max() == 0:
            main_log(
                'All intrinsics identical: W: {} H: {}, intrinsics: {}'.format(metadata_items[0].W, metadata_items[0].H,
                                                                               metadata_items[0].intrinsics))

            self._directions = get_ray_directions(metadata_items[0].W,
                                                  metadata_items[0].H,
                                                  metadata_items[0].intrinsics[0],
                                                  metadata_items[0].intrinsics[1],
                                                  metadata_items[0].intrinsics[2],
                                                  metadata_items[0].intrinsics[3],
                                                  center_pixels,
                                                  device).view(-1, 3)
        else:
            main_log('Differing intrinsics')
            self._directions = None

        append_arrays = self._check_existing_paths(chunk_paths, center_pixels, scale_factor,
                                                   len(metadata_items))
        if append_arrays is not None:
            main_log('Reusing {} chunks from previous run'.format(len(append_arrays[0])))
            self._rgb_arrays = append_arrays[0]
            self._ray_arrays = append_arrays[1]
            self._img_arrays = append_arrays[2]
        else:
            self._rgb_arrays = []
            self._ray_arrays = []
            self._img_arrays = []
            self._write_chunks(metadata_items, center_pixels, device, chunk_paths, num_chunks, scale_factor,
                               disk_flush_size)

        self._rgb_arrays.sort(key=lambda x: x.name)
        self._ray_arrays.sort(key=lambda x: x.name)
        self._img_arrays.sort(key=lambda x: x.name)

        if shuffle_chunk:
            process_log("Shuffle chunk")
            chunk_indices = list(range(len(self._rgb_arrays)))
            random.shuffle(chunk_indices)
            process_log(f"Using chunk order:")
            process_log(chunk_indices)
            self._chunk_index = cycle(chunk_indices)
        else:
            self._chunk_index = cycle(range(len(self._rgb_arrays)))
        self._loaded_rgbs = None
        self._loaded_rays = None
        self._loaded_image_indices = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
        self._chosen = None

    def load_chunk(self) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        self._chosen = chosen
        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
    
    def load_chunk_chosen(self, chosen_) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        main_log(f"Loaded {chosen}")
        while str(chosen) != chosen_:
            next_chunk_index = next(self._chunk_index)
            chosen = self._rgb_arrays[next_chunk_index]
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        main_log(f"Loaded {chosen}")
        self._chosen = chosen

        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))

    def get_state(self) -> str:
        return self._chosen

    def set_state(self, chosen: str) -> None:
        # while self._chosen != chosen:
        self.load_chunk_chosen(chosen)

    def __len__(self) -> int:
        return self._loaded_rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._loaded_rgbs[idx],
            'rays': self._loaded_rays[idx],
            'image_indices': self._loaded_image_indices[idx]
        }

    def _load_chunk_inner(self, next_index) -> Tuple[
        str, torch.FloatTensor, torch.FloatTensor, torch.ShortTensor]:
        if 'RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        # next_index = next(self._chunk_index)
        chosen = self._rgb_arrays[next_index]
        loaded_img_indices = torch.ShortTensor(np.load(self._img_arrays[next_index]))

        if self._directions is not None:
            loaded_pixel_indices = torch.IntTensor(np.load(self._ray_arrays[next_index]))

            loaded_rays = []
            for i in range(0, loaded_pixel_indices.shape[0], RAY_CHUNK_SIZE):
                image_indices = loaded_img_indices[i:i + RAY_CHUNK_SIZE]
                unique_img_indices, inverse_img_indices = torch.unique(image_indices, return_inverse=True)
                c2ws = self._c2ws[unique_img_indices.long()].to(self._device)

                pixel_indices = loaded_pixel_indices[i:i + RAY_CHUNK_SIZE]
                unique_pixel_indices, inverse_pixel_indices = torch.unique(pixel_indices, return_inverse=True)

                # (#unique images, w*h, 8)
                image_rays = get_rays_batch(self._directions[unique_pixel_indices.long()],
                                            c2ws, self._near, self._far,
                                            self._ray_altitude_range).cpu()

                del c2ws

                loaded_rays.append(image_rays[inverse_img_indices, inverse_pixel_indices])

            loaded_rays = torch.cat(loaded_rays)
        else:
            loaded_rays = torch.FloatTensor(np.load(self._ray_arrays[next_index]))

        return str(chosen), torch.FloatTensor(np.load(chosen)) / 255., loaded_rays, loaded_img_indices

    def _write_chunks(self, metadata_items: List[ImageMetadata], center_pixels: bool, device: torch.device,
                      chunk_paths: List[Path], num_chunks: int, scale_factor: int, disk_flush_size: int) -> None:
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        path_frees = []
        total_free = 0

        for chunk_path in chunk_paths:
            (chunk_path / 'rgb-chunks').mkdir(parents=True)
            (chunk_path / 'ray-chunks').mkdir(parents=True)
            (chunk_path / 'img-chunks').mkdir(parents=True)

            _, _, free = shutil.disk_usage(chunk_path)
            total_free += free
            path_frees.append(free)

        rgb_append_arrays = []
        ray_append_arrays = []
        img_append_arrays = []

        index = 0
        for chunk_path, path_free in zip(chunk_paths, path_frees):
            allocated = int(path_free / total_free * num_chunks)
            main_log('Allocating {} chunks to dataset path {}'.format(allocated, chunk_path))
            for j in range(allocated):
                rgb_array_path = chunk_path / 'rgb-chunks' / '{}.npy'.format(index)
                self._rgb_arrays.append(rgb_array_path)
                rgb_append_arrays.append(NpyAppendArray(str(rgb_array_path)))

                ray_array_path = chunk_path / 'ray-chunks' / '{}.npy'.format(index)
                self._ray_arrays.append(ray_array_path)
                ray_append_arrays.append(NpyAppendArray(str(ray_array_path)))

                img_array_path = chunk_path / 'img-chunks' / '{}.npy'.format(index)
                self._img_arrays.append(img_array_path)
                img_append_arrays.append(NpyAppendArray(str(img_array_path)))
                index += 1
        main_log('{} chunks allocated'.format(index))

        write_futures = []
        rgbs = []
        rays = []
        indices = []
        in_memory_count = 0

        if self._directions is not None:
            all_pixel_indices = torch.arange(self._directions.shape[0], dtype=torch.int)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for metadata_item in main_tqdm(metadata_items):
                image_data = get_rgb_index_mask(metadata_item)

                if image_data is None:
                    continue

                image_rgbs, image_indices, image_keep_mask = image_data
                rgbs.append(image_rgbs)
                indices.append(image_indices)
                in_memory_count += len(image_rgbs)

                if self._directions is not None:
                    image_pixel_indices = all_pixel_indices
                    if image_keep_mask is not None:
                        image_pixel_indices = image_pixel_indices[image_keep_mask == True]

                    rays.append(image_pixel_indices)
                else:
                    directions = get_ray_directions(metadata_item.W,
                                                    metadata_item.H,
                                                    metadata_item.intrinsics[0],
                                                    metadata_item.intrinsics[1],
                                                    metadata_item.intrinsics[2],
                                                    metadata_item.intrinsics[3],
                                                    center_pixels,
                                                    device)
                    image_rays = get_rays(directions, metadata_item.c2w.to(device), self._near, self._far,
                                          self._ray_altitude_range).view(-1, 8).cpu()

                    if image_keep_mask is not None:
                        image_rays = image_rays[image_keep_mask == True]

                    rays.append(image_rays)

                if in_memory_count >= disk_flush_size:
                    for write_future in write_futures:
                        write_future.result()

                    write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                        rgb_append_arrays, ray_append_arrays, img_append_arrays)

                    rgbs = []
                    rays = []
                    indices = []
                    in_memory_count = 0

            for write_future in write_futures:
                write_future.result()

            if in_memory_count > 0:
                write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                    rgb_append_arrays, ray_append_arrays, img_append_arrays)

                for write_future in write_futures:
                    write_future.result()
        for chunk_path in chunk_paths:
            chunk_metadata = {
                'images': len(metadata_items),
                'scale_factor': scale_factor
            }

            if self._directions is None:
                chunk_metadata['near'] = self._near
                chunk_metadata['far'] = self._far
                chunk_metadata['center_pixels'] = center_pixels
                chunk_metadata['ray_altitude_range'] = self._ray_altitude_range

            torch.save(chunk_metadata, chunk_path / 'metadata.pt')

        for source in [rgb_append_arrays, ray_append_arrays, img_append_arrays]:
            for append_array in source:
                append_array.close()

        main_log('Finished writing chunks to dataset paths')

    def _check_existing_paths(self, chunk_paths: List[Path], center_pixels: bool, scale_factor: int, images: int) -> \
            Optional[Tuple[List[Path], List[Path], List[Path]]]:
        rgb_arrays = []
        ray_arrays = []
        img_arrays = []

        num_exist = 0
        for chunk_path in chunk_paths:
            if chunk_path.exists():
                assert (chunk_path / 'metadata.pt').exists(), \
                    "Could not find metadata file (did previous writing to this directory not complete successfully?)"
                dataset_metadata = torch.load(chunk_path / 'metadata.pt', map_location='cpu')
                assert dataset_metadata['images'] == images
                assert dataset_metadata['scale_factor'] == scale_factor

                if self._directions is None:
                    assert dataset_metadata['near'] == self._near
                    assert dataset_metadata['far'] == self._far
                    assert dataset_metadata['center_pixels'] == center_pixels

                    if self._ray_altitude_range is not None:
                        assert (torch.allclose(torch.FloatTensor(dataset_metadata['ray_altitude_range']),
                                               torch.FloatTensor(self._ray_altitude_range)))
                    else:
                        assert dataset_metadata['ray_altitude_range'] is None

                for child in list((chunk_path / 'rgb-chunks').iterdir()):
                    rgb_arrays.append(child)
                    ray_arrays.append(child.parent.parent / 'ray-chunks' / child.name)
                    img_arrays.append(child.parent.parent / 'img-chunks' / child.name)
                num_exist += 1

        if num_exist > 0:
            assert num_exist == len(chunk_paths)
            return rgb_arrays, ray_arrays, img_arrays
        else:
            return None

    @staticmethod
    def _write_to_disk(executor: ThreadPoolExecutor, rgbs: torch.Tensor, rays: torch.FloatTensor,
                       image_indices: torch.Tensor, rgb_append_arrays, ray_append_arrays, img_append_arrays):
        indices = torch.randperm(rgbs.shape[0])
        num_chunks = len(rgb_append_arrays)
        chunk_size = math.ceil(rgbs.shape[0] / num_chunks)

        futures = []

        def append(index: int) -> None:
            rgb_append_arrays[index].append(rgbs[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            ray_append_arrays[index].append(rays[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            img_append_arrays[index].append(image_indices[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())

        for i in range(num_chunks):
            future = executor.submit(append, i)
            futures.append(future)

        return futures

# class RandomPointSampler3D:
#     def __init__(
#         self,
#         coordinates: torch.Tensor,
#         data: torch.Tensor,

#         n_points_per_sampling: int,
#     ) -> None:
#         self.n_points_per_sampling = n_points_per_sampling
#         self.flattened_coordinates = rearrange(coordinates, "d h w c-> (d h w) c")
#         self.flattened_data = rearrange(data, "d h w c-> (d h w) c")
#         # self.flattened_weight_map = rearrange(weight_map, "d h w c-> (d h w) c")
#         self.n_total_points = self.flattened_data.shape[0]

#     def next(
#         self,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         sampled_idxs = torch.randint(
#             0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
#         )# zjc 'cuda'
#         sampled_coords = self.flattened_coordinates[sampled_idxs, :]
#         sampled_data = self.flattened_data[sampled_idxs, :]
#         # sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
#         return sampled_coords, sampled_data

def denoise(
    data: np.ndarray,
    denoise_level: int,
    denoise_close: Union[bool, List[int]],
) -> np.ndarray:
    denoised_data = np.copy(data)
    if denoise_close == False:
        # using 'denoise_level' as a hard threshold,
        # the pixel with instensity below this threshold will be set to zero
        denoised_data[data <= denoise_level] = 0
    else:
        # using 'denoise_level' as a soft threshold,
        # only the pixel with itself and neighbors instensities below this threshold will be set to zero
        denoised_data[
            ndimage.binary_opening(
                data <= denoise_level,
                structure=np.ones(tuple(list(denoise_close) + [1])),
                iterations=1,
            )
        ] = 0
    return denoised_data

@dataclass
class SideInfos3D:
    dtype: str = ""
    depth: int = 0
    height: int = 0
    width: int = 0
    original_min: int = 0
    original_max: int = 0
    normalized_min: int = 0
    normalized_max: int = 0

@dataclass
class SideInfos4D:
    dtype: str = ""
    time: int = 0
    depth: int = 0
    height: int = 0
    width: int = 0
    original_min: int = 0
    original_max: int = 0
    normalized_min: int = 0
    normalized_max: int = 0

def normalize(
    data: np.ndarray, 
    sideinfos: Union[SideInfos3D, SideInfos4D]
) -> np.ndarray:
    """
    use minmax normalization to scale and offset the data range to [normalized_min,normalized_max]
    """
    normalized_min, normalized_max = sideinfos.normalized_min, sideinfos.normalized_max
    dtype = data.dtype.name
    data = data.astype(np.float32)
    original_min = float(data.min())
    original_max = float(data.max())
    data = (data - original_min) / (original_max - original_min)
    data *= normalized_max - normalized_min
    data += normalized_min
    sideinfos.dtype = dtype
    sideinfos.original_min = original_min
    sideinfos.original_max = original_max
    return data

def inv_normalize(
    data: np.ndarray, 
    sideinfos: Union[SideInfos3D, SideInfos4D]
) -> np.ndarray:
    dtype = sideinfos.dtype
    if dtype == "uint8":
        dtype = np.uint8
    elif dtype == "uint12":
        dtype = np.uint12
    elif dtype == "uint16":
        dtype = np.uint16
    elif dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    else:
        raise NotImplementedError
    data -= sideinfos.normalized_min
    data /= sideinfos.normalized_max - sideinfos.normalized_min
    data = np.clip(data, 0, 1)
    data = (
        data * (sideinfos.original_max - sideinfos.original_min)
        + sideinfos.original_min
    )
    data = np.array(data, dtype=dtype)
    return data

sideinfos = SideInfos3D()
sideinfos.normalized_min = 0
sideinfos.normalized_max = 100

class Mydata(Dataset):
    def __init__(self, hparams) -> None:
        super(Mydata, self).__init__()
        #读取data --> denoise --> normalize
        self.data_path = hparams.data_path
        self.data = tifffile.imread(self.data_path)
        if len(self.data.shape) == 3:
            self.data = self.data[..., None]
        assert (
            len(self.data.shape) == 4
        ), "Only DHWC data is allowed. Current data shape is {}.".format(self.data.shape)
        
        self.denoise_data = denoise(self.data, denoise_level=0, denoise_close= [2,2,2] )
        self.normalized_data = normalize(self.denoise_data, sideinfos)

        #生成坐标
        sideinfos.coord_normalized_min = -1
        sideinfos.coord_normalized_max = 1
        sideinfos.depth, sideinfos.height, sideinfos.width, _ = self.data.shape
        self.coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(sideinfos.coord_normalized_min, sideinfos.coord_normalized_max, sideinfos.depth),
                torch.linspace(
                    sideinfos.coord_normalized_min, sideinfos.coord_normalized_max, sideinfos.height
                ),
                torch.linspace(sideinfos.coord_normalized_min, sideinfos.coord_normalized_max, sideinfos.width),
                indexing="ij",
            ),
            axis=-1,
        )
        # self.sampler = RandomPointSampler3D(
        #     self.coordinates, self.normalized_data, hparams.n_random_training_samples
        # )
        self.normalized_data = rearrange(self.normalized_data, 'd h w c -> (d h w) c')
        self.coordinates =  rearrange(self.coordinates, 'd h w c -> (d h w) c')

    def __len__(self):
        return self.normalized_data.shape[0]
        
    def __getitem__(self, idx):
        coords, gt = self.coordinates[idx,:], self.normalized_data[idx, :]
        input = {
            'coords': coords,
            'gt': gt
        }
        return input

        
