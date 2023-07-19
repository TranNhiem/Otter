import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


import ast
import functools
import io
import json
import logging
import math
import os
import random
import sys
import tarfile
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import torch
import torch.utils
import torchvision
import webdataset as wds
from PIL import Image, ImageSequence, ImageFile
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample
from pipeline.mimicit_utils.mimicit_dataset import MimicitDataset

import statistics

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) if os.path.basename(shard) in sizes else 0 for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and ("png" in sample or "jpg" in sample or "jpeg" in sample)


def decode_base64_image(key, value):
    if not key.endswith(".png"):
        return None
    rawbytes = base64.b64decode(value)
    image = Image.open(io.BytesIO(rawbytes))
    # Check if the image is in palette mode and has transparency
    if image.mode == "P":
        try:
            alpha = image.getchannel("A")
            if alpha.mode == "L":
                image = image.convert("RGBA")
        except ValueError:
            pass
    image = image.convert("RGB")
    return image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(exn):  # Avoid spamming logs with these
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def preprocess_image(sample, image_processor):
    image = [image_processor.preprocess(s, return_tensors="pt")["pixel_values"] for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip wo/w color jitter
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # image = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(image)
    return image


def preprocess_text(sample, tokenizer):
    tokenizer.padding_side = "right"
    sample = [(f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample]
    text = tokenizer(
        sample,
        max_length=32,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


MIN_KB = 10
MAX_NUM_IMAGES = 5
import base64


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    info = json.loads(sample[0])
    sentences = info["text_list"]

    images, sentence_ixs = [], []

    for sample_image in info["image_info"]:
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sample_image["matched_sim"] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes))

        # Check if the image is in palette mode and has transparency
        if image.mode == "P" and "transparency" in image.info:
            try:
                image = image.convert("RGBA")
            except ValueError:
                pass

        image = image.convert("RGB")

        images.append(image)
        sentence_ixs.append(sample_image["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # images -> tensors
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros((MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float)
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = text.replace(" <|endofchunk|>", "<|endofchunk|>").replace("<image> ", "<image>").replace(" <image>", "<image>")
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(text_tensor["input_ids"] == tokenizer.additional_special_tokens_ids[tokenizer.additional_special_tokens.index("<image>")])

    if num_images == 0:
        raise ValueError("No images in sample")
    elif num_images == 1 and random.random() <= 0.5:  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_mmc4_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.mmc4_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_mmc4
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_interleaved,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_mmc4, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_mmc4 * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.laion_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_text, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode(decode_base64_image, only="png", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
            wds.map_tuple(preprocess_image_fn, preprocess_text_fn, handler=log_and_continue),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


import json

from PIL import Image, ImageFile

from pipeline.mimicit_utils.mimicit_dataset import MimicitDataset


def get_mimicit_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    args.task = "pretrain"
    args.tokenizer = tokenizer
    unified_datasets = []
    # processing for image-text in-context datasets
    if args.mimicit_ic_path != "":
        all_mimicit_ic_path = (
            args.mimicit_ic_path.split(",") + args.past_mimicit_ic_path.split(",") if args.past_mimicit_ic_path != "" else args.mimicit_ic_path.split(",")
        )
        all_images_ic_path = (
            args.images_ic_path.split(",") + args.past_images_ic_path.split(",") if args.past_images_ic_path != "" else args.images_ic_path.split(",")
        )
        all_train_config_ic_path = (
            args.train_config_ic_path.split(",") + args.past_train_config_ic_path.split(",")
            if args.past_train_config_ic_path != ""
            else args.train_config_ic_path.split(",")
        )
        if args.past_mimicit_ic_path != "":
            ic_status = ["new"] * len(args.mimicit_ic_path.split(",")) + ["past"] * len(args.past_mimicit_ic_path.split(","))
        else:
            ic_status = ["new"] * len(args.mimicit_ic_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_ic_path, all_images_ic_path, all_train_config_ic_path, status_list=ic_status)
        unified_datasets.append(unified_dataset)

    # processing for image-text datasets
    if args.mimicit_path != "":
        all_mimicit_path = args.mimicit_path.split(",") + args.past_mimicit_path.split(",") if args.past_mimicit_path != "" else args.mimicit_path.split(",")
        all_images_path = args.images_path.split(",") + args.past_images_path.split(",") if args.past_images_path != "" else args.images_path.split(",")
        if args.past_mimicit_path != "":
            status = ["new"] * len(args.mimicit_path.split(",")) + ["past"] * len(args.past_mimicit_path.split(","))
        else:
            status = ["new"] * len(args.mimicit_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_path, all_images_path, status_list=status)
        unified_datasets.append(unified_dataset)

    # processing for text datasets
    if args.mimicit_text_path != "":
        all_mimicit_text_path = (
            args.mimicit_text_path.split(",") + args.past_mimicit_text_path.split(",")
            if args.past_mimicit_text_path != ""
            else args.mimicit_text_path.split(",")
        )
        if args.past_mimicit_text_path != "":
            text_status = ["new"] * len(args.mimicit_text_path.split(",")) + ["past"] * len(args.past_mimicit_text_path.split(","))
        else:
            text_status = ["new"] * len(args.mimicit_text_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_text_path, status_list=text_status)
        unified_datasets.append(unified_dataset)

    # processing for video-text datasets
    if args.mimicit_vt_path != "":
        all_mimicit_vt_path = (
            args.mimicit_vt_path.split(",") + args.past_mimicit_vt_path.split(",") if args.past_mimicit_vt_path != "" else args.mimicit_vt_path.split(",")
        )
        all_images_vt_path = (
            args.images_vt_path.split(",") + args.past_images_vt_path.split(",") if args.past_images_vt_path != "" else args.images_vt_path.split(",")
        )
        if args.past_mimicit_vt_path != "":
            vt_status = ["new"] * len(args.mimicit_vt_path.split(",")) + ["past"] * len(args.past_mimicit_vt_path.split(","))
        else:
            vt_status = ["new"] * len(args.mimicit_vt_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_vt_path, all_images_vt_path, status_list=vt_status)
        unified_datasets.append(unified_dataset)

    # args.train_num_samples = sum(len(dataset) for dataset in unified_datasets) / len(unified_datasets)
    args.train_num_samples = statistics.median((len(dataset) for dataset in unified_datasets))
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size

    num_samples = args.train_num_samples  # 8
    num_batches = round_fn(num_samples / global_batch_size)  # 2
    # args.workers = max(1, args.workers)  # 1
    # num_worker_batches = round_fn(num_batches / args.workers)  # per dataloader worker #2
    # num_batches = num_worker_batches * args.workers  # 2
    num_samples = num_batches * global_batch_size  # 8

    dataloaders = []

    # unified_datasets = unified_old_datasets + unified_new_datasets

    for unified_dataset in unified_datasets:
        sampler = RandomSampler(unified_dataset, replacement=True, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(
            unified_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unified_dataset.collate,
        )

        dataloaders.append(dataloader)
    return dataloaders


def get_mimicit_dataset_continual(args, image_processor, tokenizer, epoch=0, floor=False, subset=300):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    args.task = "pretrain"
    args.tokenizer = tokenizer
    unified_datasets = []
    # processing for image-text in-context datasets
    if args.mimicit_ic_path != "":
        all_mimicit_ic_path = (
            args.mimicit_ic_path.split(",") + args.past_mimicit_ic_path.split(",") if args.past_mimicit_ic_path != "" else args.mimicit_ic_path.split(",")
        )
        all_images_ic_path = (
            args.images_ic_path.split(",") + args.past_images_ic_path.split(",") if args.past_images_ic_path != "" else args.images_ic_path.split(",")
        )
        all_train_config_ic_path = (
            args.train_config_ic_path.split(",") + args.past_train_config_ic_path.split(",")
            if args.past_train_config_ic_path != ""
            else args.train_config_ic_path.split(",")
        )
        if args.past_mimicit_ic_path != "":
            ic_status = ["new"] * len(args.mimicit_ic_path.split(",")) + ["past"] * len(args.past_mimicit_ic_path.split(","))
        else:
            ic_status = ["new"] * len(args.mimicit_ic_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_ic_path, all_images_ic_path, all_train_config_ic_path, status_list=ic_status)
        unified_datasets.append(unified_dataset)

    # processing for image-text datasets
    if args.mimicit_path != "":
        all_mimicit_path = args.mimicit_path.split(",") + args.past_mimicit_path.split(",") if args.past_mimicit_path != "" else args.mimicit_path.split(",")
        all_images_path = args.images_path.split(",") + args.past_images_path.split(",") if args.past_images_path != "" else args.images_path.split(",")
        if args.past_mimicit_path != "":
            status = ["new"] * len(args.mimicit_path.split(",")) + ["past"] * len(args.past_mimicit_path.split(","))
        else:
            status = ["new"] * len(args.mimicit_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_path, all_images_path, status_list=status)
        unified_datasets.append(unified_dataset)

    # processing for text datasets
    if args.mimicit_text_path != "":
        all_mimicit_text_path = (
            args.mimicit_text_path.split(",") + args.past_mimicit_text_path.split(",")
            if args.past_mimicit_text_path != ""
            else args.mimicit_text_path.split(",")
        )
        if args.past_mimicit_text_path != "":
            text_status = ["new"] * len(args.mimicit_text_path.split(",")) + ["past"] * len(args.past_mimicit_text_path.split(","))
        else:
            text_status = ["new"] * len(args.mimicit_text_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_text_path, status_list=text_status)
        unified_datasets.append(unified_dataset)

    # processing for video-text datasets
    if args.mimicit_vt_path != "":
        all_mimicit_vt_path = (
            args.mimicit_vt_path.split(",") + args.past_mimicit_vt_path.split(",") if args.past_mimicit_vt_path != "" else args.mimicit_vt_path.split(",")
        )
        all_images_vt_path = (
            args.images_vt_path.split(",") + args.past_images_vt_path.split(",") if args.past_images_vt_path != "" else args.images_vt_path.split(",")
        )
        if args.past_mimicit_vt_path != "":
            vt_status = ["new"] * len(args.mimicit_vt_path.split(",")) + ["past"] * len(args.past_mimicit_vt_path.split(","))
        else:
            vt_status = ["new"] * len(args.mimicit_vt_path.split(","))
        unified_dataset = MimicitDataset(args, all_mimicit_vt_path, all_images_vt_path, status_list=vt_status)
        unified_datasets.append(unified_dataset)

    # args.train_num_samples = sum(len(dataset) for dataset in unified_datasets) / len(unified_datasets)
    args.train_num_samples = statistics.median((len(dataset) for dataset in unified_datasets))
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size

    num_samples = args.train_num_samples  # 8
    num_batches = round_fn(num_samples / global_batch_size)  # 2
    # args.workers = max(1, args.workers)  # 1
    # num_worker_batches = round_fn(num_batches / args.workers)  # per dataloader worker #2
    # num_batches = num_worker_batches * args.workers  # 2
    num_samples = num_batches * global_batch_size  # 8

    dataloaders = []

    # unified_datasets = unified_old_datasets + unified_new_datasets

    for unified_dataset in unified_datasets:
        subset_data = torch.utils.data.Subset(unified_dataset, range(0, subset))
        print("subset size:", len(subset_data))
        # sampler = RandomSampler(subset_data, replacement=True, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(
            subset_data,
            # sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unified_dataset.collate,
        )

        dataloaders.append(dataloader)
    return [dataloaders, unified_dataset.collate]


def get_dataset_fn(dataset_type):
    if dataset_type == "laion":
        return get_laion_dataset
    elif dataset_type == "mmc4":
        return get_mmc4_dataset
    elif dataset_type == "mimicit":
        # return get_mimicit_dataset
        return get_mimicit_dataset_continual
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    return get_dataset_fn(dataset_type)(args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)


def get_continual_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    import copy

    train_datasets = []
    test_datasets = []

    current_args = copy.deepcopy(args)
    # get the first vqa data  --- st-vqa
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_train_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_train.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_train_train.json"
    train_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_test_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_test.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/st-vqa/st-vqa_original_test_train.json"
    test_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])
    # get the second vqa data --- ok-vqa
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_train_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_train.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_train_train.json"
    train_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_val_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_val.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/okvqa/okvqa_original_val_train.json"
    test_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])
    # get the third vqa data --- gqa
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_train_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_train.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_train_train.json"
    train_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])
    current_args.mimicit_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_val_instructions.json"
    current_args.images_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_val.json"
    current_args.train_config_ic_path = "/mnt/lustre/dwzhou/data/gpt_split/m3it/vqa/gqa/gqa_original_val_train.json"
    test_datasets.append(get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[0])

    collate = get_dataset_fn(dataset_type)(current_args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer)[1]
    train_datasets = host_replay(train_datasets, args, subset_size=100, collate=collate)

    return train_datasets, test_datasets


def host_replay(train_datasets, args, subset_size=100, collate=None):
    from torch.utils.data import Subset, ConcatDataset

    new_train_data = []
    # for all training datasets, get the subset from the prior datasets and concat them together.
    pure_train_dataset = [train_datasets[i][0].dataset for i in range(len(train_datasets))]
    for i in range(len(train_datasets) - 1, -1, -1):
        if i > 0:
            for j in range(i):
                pure_train_dataset[i] = ConcatDataset([pure_train_dataset[i], Subset(pure_train_dataset[j], range(0, subset_size))])
        dataloader = torch.utils.data.DataLoader(
            pure_train_dataset[i],
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            # collate_fn=train_datasets[0][0].dataset.collate,
            collate_fn=collate,
        )
        new_train_data.append(dataloader)

    # reverse the order of new_train_data
    new_train_data = new_train_data[::-1]
    for i in range(len(new_train_data)):
        print(len(new_train_data[i].dataset))
        new_train_data[i] = [new_train_data[i]]

    return new_train_data


class DataManager(object):
    def __init__(self, args):
        self.args = args

        self.dataset_name = args.dataset
        self.data_seed = args.data_seed
        self.shuffle = True

        self._setup_data()
        self._increments = np.ones(self._nb_tasks)  # each time learn a new task

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[: task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(appendent_data, appendent_targets, low_range=idx, high_range=idx + 1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), DummyDataset(val_data, val_targets, trsf, self.use_path)

    # def _setup_data(self, dataset_name, shuffle, seed):
    #     idata = _get_idata(dataset_name)
    #     idata.download_data()

    #     # Data
    #     self._train_data, self._train_targets = idata.train_data, idata.train_targets
    #     self._test_data, self._test_targets = idata.test_data, idata.test_targets
    #     self.use_path = idata.use_path

    #     # Transforms
    #     self._train_trsf = idata.train_trsf
    #     self._test_trsf = idata.test_trsf
    #     self._common_trsf = idata.common_trsf

    #     # Order
    #     order = [i for i in range(len(np.unique(self._train_targets)))]
    #     if shuffle:
    #         np.random.seed(seed)
    #         order = np.random.permutation(len(order)).tolist()
    #     else:
    #         order = idata.class_order
    #     self._class_order = order
    #     logging.info(self._class_order)

    #     # Map indices
    #     self._train_targets = _map_new_class_index(
    #         self._train_targets, self._class_order
    #     )
    #     self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _setup_data(self):
        self.all_data_names = []
        self.all_train_loaders = []
        self.all_test_loaders = []
        if self.dataset_name == "vqa":
            # a-okvqa  gqa  okvqa  st-vqa  text-vqa  vqav2
            self.all_data_names = ["st-vqa", "okvqa"]
            mimicit_loaders = get_data(args, image_processor, tokenizer, "mimicit")
            for i in range(len(self.all_data_names)):
                self.all_train_loaders.append(mimicit_loaders[i][0])
                self.all_test_loaders.append(mimicit_loaders[i][1])
            # append data loaders for each data split

        # idata = _get_idata(dataset_name)
        # idata.download_data()

        # # Data
        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        # self.use_path = idata.use_path

        # # Transforms
        # self._train_trsf = idata.train_trsf
        # self._test_trsf = idata.test_trsf
        # self._common_trsf = idata.common_trsf

        # # Order
        # order = [i for i in range(len(np.unique(self._train_targets)))]
        # if shuffle:
        #     np.random.seed(seed)
        #     order = np.random.permutation(len(order)).tolist()
        # else:
        #     order = idata.class_order
        # self._class_order = order
        # logging.info(self._class_order)

        # # Map indices
        # self._train_targets = _map_new_class_index(
        #     self._train_targets, self._class_order
        # )
        # self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
