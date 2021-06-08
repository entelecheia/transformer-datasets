# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Korean spacing recognition dataset"""

import datasets
from smart_open import open

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Lee:2021,
  title     = "Korean Spacing Recognition Dataset",
  authors   = "Young Joon Lee",
  publisher = "GitHub",
  year      = "2021"
}
"""


_DESCRIPTION = """\
Korean spacing recognition dataset
"""

_HOMEPAGE = "https://github.com/entelecheia/ekonspacing"

_LICENSE = "MIT License for non-commercial use"

_URL = "https://www.dropbox.com/s/"
_TRAINING_FILE = "uj2f5ouxwoojfb9/train.txt?dl=1"
_DEV_FILE = "mmcxwiucc6ge7df/val.txt?dl=1"
_TEST_FILE = "jjqauzmzb5oaogv/test.txt?dl=1"
_SMALL_TRAINING_FILE = "1opfrej3b9dy04d/train_small.txt?dl=1"
_SMALL_DEV_FILE = "didx1np8e31ak3i/val_small.txt?dl=1"
_SMALL_TEST_FILE = "5me24mvgk0n61cx/test_small.txt?dl=1"


_DEFAULT = "default"

_VERSION = "1.0.0"

_TAGS = ["I", "O", "B", "E", "S"]


def _get_tags(words):
    tags = []
    for word in words:
        if len(word) == 1:
            tags.append(_TAGS.index("S"))
        elif len(word) > 1:
            for i, c in enumerate(word):
                if i == 0:
                    tags.append(_TAGS.index("B"))
                elif i == len(word) - 1:
                    tags.append(_TAGS.index("E"))
                else:
                    tags.append(_TAGS.index("I"))
    return tags


class eKonSpacingConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for eKonSpacing.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(eKonSpacingConfig, self).__init__(**kwargs)


class eKonSpacing(datasets.GeneratorBasedBuilder):
    """Korean spacing recognition dataset"""

    BUILDER_CONFIG_CLASS = eKonSpacingConfig

    BUILDER_CONFIGS = [
        eKonSpacingConfig(
            name="default",
            version=datasets.Version(_VERSION, ""),
            description="Korean spacing recognition dataset",
        ),
        eKonSpacingConfig(
            name="small",
            version=datasets.Version(_VERSION, ""),
            description="Korean spacing recognition small dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = _DEFAULT

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "spacing_tags": datasets.Sequence(datasets.features.ClassLabel(names=_TAGS)),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "small":
            urls_to_download = {
                "train": f"{_URL}{_SMALL_TRAINING_FILE}",
                "validation": f"{_URL}{_SMALL_DEV_FILE}",
                "test": f"{_URL}{_SMALL_TEST_FILE}",
            }
        else:
            urls_to_download = {
                "train": f"{_URL}{_TRAINING_FILE}",
                "validation": f"{_URL}{_DEV_FILE}",
                "test": f"{_URL}{_TEST_FILE}",
            }

        if self.config.data_files:
            downloaded_files = self.config.data_files
        else:
            downloaded_files = dl_manager.download(urls_to_download)
        print(downloaded_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        print("⏳ Generating examples from = {}".format(filepath))
        with open(filepath, encoding="utf-8") as f:
            words = ""
            spacing_tags = []
            for id_, row in enumerate(f):
                if id_ < 5:
                    print(row)
                row = row.strip()
                if row:
                    words = row.split()
                    spacing_tags = _get_tags(words)
                    yield id_, {
                        "text": row,
                        "tokens": list("".join(words)),
                        "spacing_tags": spacing_tags,
                    }
