import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from ._base import BaseModel, module_registry
from ..util.misc import get_root_folder

from lmdeploy import pipeline, TurbomindEngineConfig


class InternVLBase(BaseModel):
    model_name = "[MODEL_NAME]"

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)
        self.model_name_id = self.model_name.split("/")[-1]
        path = get_root_folder() / "pretrained_models" / "internvl2" / self.model_name_id
        if not path.exists():
            self.prepare()
        backend_config = TurbomindEngineConfig(cache_max_entry_count=0.05)
        self.pipe = pipeline(str(path), backend_config=backend_config)

    @torch.no_grad()
    def forward(self, input_image, query, *_, **__):
        image = T.functional.to_pil_image(input_image)
        return self.pipe((query, image)).text

    @classmethod
    def prepare(cls):
        snapshot_download(cls.model_name,
            local_dir=get_root_folder() / "pretrained_models" / "internvl2" / cls.model_name.split("/")[-1])


@module_registry.register("internvl2.5_8b")
class InternVL2_5_8B(InternVLBase):
    model_name = "OpenGVLab/InternVL2_5-8B-AWQ"
