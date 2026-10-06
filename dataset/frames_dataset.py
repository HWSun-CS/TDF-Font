import os
import glob
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
from torch.utils.data import Dataset


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and '.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = os.listdir(name)
        frames = [f for f in frames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        frames = sorted(frames)
        num_frames = len(frames)
        video_array = []
        for idx in range(num_frames):
            img = io.imread(os.path.join(name, frames[idx]))
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = gray2rgb(img)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[..., :3]
            video_array.append(img_as_float32(img))
        video_array = np.array(video_array)
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = img_as_float32(image)
        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith(('.gif', '.mp4', '.mov')):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
      - nested folders: root/font_id/(train|test)/character_name/frames.png
    """

    def __init__(
        self,
        root_dir,
        frame_shape=(256, 256, 3),
        id_sampling=False,
        is_train=True,
        pairs_list=None,
        augmentation_params=None,
        use_last_frame_as_style=False,
        font_prefix=None,
        use_mid_frame=False,
        style_from_same_font=False,
    ):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.use_last_frame_as_style = use_last_frame_as_style
        self.use_mid_frame = bool(use_mid_frame) and is_train
        self.font_prefix = font_prefix if (font_prefix is not None and str(font_prefix).strip() != '') else None
        self.is_train = is_train
        self.style_from_same_font = bool(style_from_same_font)

        self.samples: List[Dict[str, str]] = []

        def _apply_font_filter(names: List[Path]) -> List[Path]:
            if self.font_prefix is None:
                return names
            return [n for n in names if str(n.name).startswith(self.font_prefix)]

        root_path = Path(root_dir)
        font_dirs = [p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
        has_nested_split = all(((p / 'train').exists() or (p / 'test').exists()) for p in font_dirs) and len(font_dirs) > 0

        if has_nested_split:
            split_name = 'train' if is_train else 'test'
            font_dirs = _apply_font_filter(font_dirs)
            for font_dir in font_dirs:
                split_dir = font_dir / split_name
                if not split_dir.exists():
                    continue
                for char_dir in split_dir.iterdir():
                    if not char_dir.is_dir() or char_dir.name.startswith('.'):
                        continue
                    self.samples.append(
                        {
                            'path': str(char_dir),
                            'font_id': font_dir.name,
                            'char_name': char_dir.name,
                            'video_id': f"{font_dir.name}/{split_name}/{char_dir.name}",
                        }
                    )
            if len(self.samples) == 0 and self.font_prefix is not None:
                print(f"Warning: font_prefix='{self.font_prefix}' matched 0 folders under {root_dir}.")
        else:
            videos = [f for f in os.listdir(root_dir) if not f.startswith('.')]

            if os.path.exists(os.path.join(root_dir, 'train')):
                assert os.path.exists(os.path.join(root_dir, 'test'))
                print("Use predefined train-test split.")
                if id_sampling:
                    train_videos = {os.path.basename(video).split('#')[0] for video in os.listdir(os.path.join(root_dir, 'train')) if not video.startswith('.')}
                    train_videos = list(train_videos)
                else:
                    train_videos = [f for f in os.listdir(os.path.join(root_dir, 'train')) if not f.startswith('.')]
                test_videos = [f for f in os.listdir(os.path.join(root_dir, 'test')) if not f.startswith('.')]

                if self.font_prefix is not None:
                    train_videos = [v for v in train_videos if str(v).startswith(self.font_prefix)]
                    test_videos = [v for v in test_videos if str(v).startswith(self.font_prefix)]
                self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            else:
                from sklearn.model_selection import train_test_split

                print("Use random train-test split.")
                base_list = [v for v in videos if (self.font_prefix is None or str(v).startswith(self.font_prefix))]
                train_videos, test_videos = train_test_split(base_list, test_size=0.2)

            chosen = train_videos if is_train else test_videos
            if self.font_prefix is not None and len(chosen) == 0:
                print(f"Warning: font_prefix='{self.font_prefix}' matched 0 folders under {self.root_dir}.")
            for name in chosen:
                font_id_val = None
                try:
                    font_id_val = int(os.path.basename(name).split(os.sep)[0])
                except Exception:
                    font_id_val = None
                self.samples.append(
                    {
                        'path': os.path.join(self.root_dir, name),
                        'font_id': font_id_val,
                        'char_name': name,
                        'video_id': name,
                    }
                )

        self.transform = None
        self.font_to_indices: Dict[int, List[int]] = {}
        for i, s in enumerate(self.samples):
            fid = s.get('font_id', None)
            if fid is None:
                continue
            self.font_to_indices.setdefault(fid, []).append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        video_name = sample.get('video_id', os.path.basename(path))

        stacked = None
        frame_order = None
        alpha = None
        style_img = None

        if self.is_train:
            if os.path.isdir(path):
                frames = os.listdir(path)
                frames = [f for f in frames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                frames = sorted(frames)
                num_frames = len(frames)

                if num_frames < 2:
                    raise ValueError(f"Expect at least 2 frames in {path}, got {num_frames}")
                source_idx = 0
                driving_idx = num_frames - 1

                def _read_frame(frame_idx):
                    img = io.imread(os.path.join(path, frames[frame_idx]))
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        img = gray2rgb(img)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = img[..., :3]
                    return img_as_float32(img)

                source_img = _read_frame(source_idx)
                driving_img = _read_frame(driving_idx)
                frames_to_stack = [source_img, driving_img]
                frame_order = ['source', 'driving']

                if self.use_mid_frame:
                    s, d = sorted([source_idx, driving_idx])
                    if d - s >= 2:
                        mid_idx = np.random.randint(s + 1, d)
                        alpha = np.float32((mid_idx - s) / float(d - s))
                        mid_img = _read_frame(mid_idx)
                        frames_to_stack.append(mid_img)
                        frame_order.append('mid')

                stacked = np.array(frames_to_stack, dtype='float32')
            else:
                video_array = read_video(path, frame_shape=self.frame_shape)
                num_frames = len(video_array)
                if num_frames < 2:
                    raise ValueError(f"Expect at least 2 frames in {path}, got {num_frames}")
                source_idx = 0
                driving_idx = num_frames - 1
                source_img = video_array[source_idx]
                driving_img = video_array[driving_idx]
                frames_to_stack = [source_img, driving_img]
                frame_order = ['source', 'driving']

                if self.use_mid_frame:
                    s, d = sorted([source_idx, driving_idx])
                    if d - s >= 2:
                        mid_idx = np.random.randint(s + 1, d)
                        alpha = np.float32((mid_idx - s) / float(d - s))
                        frames_to_stack.append(video_array[mid_idx])
                        frame_order.append('mid')

                stacked = np.array(frames_to_stack, dtype='float32')

            if self.transform is not None and stacked is not None:
                stacked = self.transform(stacked)
            font_id = sample.get('font_id', None)
            if self.style_from_same_font and font_id is not None:
                style_idx = self._sample_same_font(font_id, idx)
                if style_idx is not None:
                    style_path = self.samples[style_idx]['path']
                    style_img = self._load_last_frame(style_path)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = range(num_frames)
            video_array = video_array[frame_idx]
            if self.transform is not None:
                video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(stacked[0], dtype='float32')
            driving = np.array(stacked[1], dtype='float32')
            template = source

            if len(source.shape) == 2:
                source = np.expand_dims(source, axis=-1)
            if len(driving.shape) == 2:
                driving = np.expand_dims(driving, axis=-1)

            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
            out['template'] = template.transpose((2, 0, 1))
            if frame_order and 'mid' in frame_order:
                mid_idx_in_stack = frame_order.index('mid')
                mid = np.array(stacked[mid_idx_in_stack], dtype='float32')
                if len(mid.shape) == 2:
                    mid = np.expand_dims(mid, axis=-1)
                out['mid'] = mid.transpose((2, 0, 1))
                out['alpha'] = np.array(alpha, dtype='float32')
            if self.use_last_frame_as_style:
                out['video_id'] = video_name
            font_id = sample.get('font_id', None)
            if font_id is not None:
                try:
                    out['font_id'] = int(font_id)
                except Exception:
                    out['font_id'] = font_id
            if style_img is not None:
                if len(style_img.shape) == 2:
                    style_img = np.expand_dims(style_img, axis=-1)
                out['style'] = style_img.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out

    def _sample_same_font(self, font_id, cur_idx):
        """
        Sample a random example from the same font
        """
        candidates = self.font_to_indices.get(font_id, None)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        filtered = [i for i in candidates if i != cur_idx]
        if len(filtered) == 0:
            filtered = candidates
        return int(np.random.choice(filtered))

    def _load_last_frame(self, path: str):
        """
        Load the last frame from a sample path
        """
        if os.path.isdir(path):
            frames = os.listdir(path)
            frames = [f for f in frames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            frames = sorted(frames)
            if len(frames) == 0:
                return None
            img = io.imread(os.path.join(path, frames[-1]))
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = gray2rgb(img)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[..., :3]
            return img_as_float32(img)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            if len(video_array) == 0:
                return None
            frame = video_array[-1]
            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                frame = gray2rgb(frame)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]
            return img_as_float32(frame)


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = getattr(self.initial_dataset, 'videos', None)
            if videos is None:
                videos = [s.get('video_id', str(i)) for i, s in enumerate(getattr(self.initial_dataset, 'samples', []))]
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}

