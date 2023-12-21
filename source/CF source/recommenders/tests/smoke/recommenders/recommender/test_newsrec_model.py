# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest

try:
    from recommenders.models.newsrec.newsrec_utils import prepare_hparams
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.models.base_model import BaseModel
    from recommenders.models.newsrec.models.nrms import NRMSModel
    from recommenders.models.newsrec.models.naml import NAMLModel
    from recommenders.models.newsrec.models.lstur import LSTURModel
    from recommenders.models.newsrec.io.mind_iterator import MINDIterator
    from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


@pytest.mark.gpu
def test_model_nrms(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", "news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", "behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", "news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", "behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", "nrms.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = NRMSModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.gpu
def test_model_naml(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", "news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", "behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", "news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", "behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(mind_resource_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(mind_resource_path, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", "naml.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        epochs=1,
    )

    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator)
    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.gpu
def test_model_lstur(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", "news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", "behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", "news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", "behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", "lstur.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = LSTURModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.gpu
def test_model_npa(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", "news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", "behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", "news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", "behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", "lstur.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = LSTURModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )
