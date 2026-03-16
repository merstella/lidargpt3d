import logging
from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.nuscenes_point_dataset import NuScenesPointCloudDataset

from nuscenes.nuscenes import NuScenes

@registry.register_builder("NuScenes_caption")
class NuScenesCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = NuScenesPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/NuScenes/nucaption.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        nusc = NuScenes(version=build_info.version, dataroot=build_info.data_root, verbose=False)

        datasets["train"] = self.train_dataset_cls(
            text_processor=self.text_processors["train"],
            nusc=nusc,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            normalize_pc=getattr(build_info, "normalize_pc", False),
            use_intensity=getattr(build_info, "use_intensity", True),
            use_time_or_ring=getattr(build_info, "use_time_or_ring", True),
            return_dim=getattr(build_info, "return_dim", 6),
            feat_mode=getattr(build_info, "feat_mode", "const_0p4"),
        )
        return datasets


@registry.register_builder("NuScenes_qa")
class NuScenesQABuilder(BaseDatasetBuilder):
    train_dataset_cls = NuScenesPointCloudDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/NuScenes/nuscenesqa.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        nusc = NuScenes(version=build_info.version, dataroot=build_info.data_root, verbose=False)

        datasets["train"] = self.train_dataset_cls(
            text_processor=self.text_processors["train"],
            nusc=nusc,
            anno_path=build_info.ann_path,
            pointnum=build_info.pointnum,
            normalize_pc=getattr(build_info, "normalize_pc", False),
            use_intensity=getattr(build_info, "use_intensity", True),
            use_time_or_ring=getattr(build_info, "use_time_or_ring", True),
            return_dim=getattr(build_info, "return_dim", 6),
            feat_mode=getattr(build_info, "feat_mode", "const_0p4"),
        )
        return datasets
