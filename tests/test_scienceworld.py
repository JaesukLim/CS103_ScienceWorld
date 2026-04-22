import json

import numpy as np
import pytest

import cs103_scienceworld.scienceworld as scienceworld_module
from cs103_scienceworld import CS103ScienceWorldFinalProjectEnv, ScienceWorldEnv
from cs103_scienceworld.constants import TASKS, VISIBLE_ID2TASK


def test_observation_is_deterministic():
    env = ScienceWorldEnv("1-1")
    obs_orig, _ = env.reset()

    for _ in range(30):
        obs, _ = env.reset()
        assert obs == obs_orig

        obs, _, _, _ = env.step("look around")
        assert obs == obs_orig


def test_multiple_instances():
    env1 = ScienceWorldEnv("1-1")
    env2 = ScienceWorldEnv("1-1")

    assert env1._gateway._gateway_client.port != env2._gateway._gateway_client.port

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    # Check if the two observations are the same when ignoring the order in which objects are described.
    assert obs1 == obs2

    # Interact with one of the envs.
    env1.step("open door to art studio")

    # Check if the observations now differ from each other.
    obs1_1, _, _, _ = env1.step("look around")
    obs2_1, _, _, _ = env2.step("look around")
    assert obs1_1 != obs2_1

    # Resetting the second env doesn't affect the first one.
    env2.reset()

    obs1_2, _, _, _ = env1.step("look around")
    assert obs1_1 == obs1_2

    env2.step("open door to art studio")
    obs2_2, _, _, _ = env2.step("look around")
    assert obs1_2 == obs2_2


def test_closing_env():
    env = ScienceWorldEnv()
    env.task_names  # Load task names.
    assert env._gateway.java_process.poll() is None
    env.close()
    env._gateway.java_process.wait(5)
    assert env._gateway.java_process.poll() is not None


def test_variation_sets_are_disjoint():
    env = ScienceWorldEnv()

    for task in env.get_task_names():
        env.load(task)
        train = set(env.get_variations_train())
        dev = set(env.get_variations_dev())
        test = set(env.get_variations_test())
        assert set.isdisjoint(train, dev)
        assert set.isdisjoint(train, test)
        assert set.isdisjoint(dev, test)


def test_load():
    # Test loading former task names.
    FORMER_TASK_NAMES = {
        "1-1": "task-1-boil",
        "1-2": "task-1-melt",
        "1-3": "task-1-freeze",
        "1-4": "task-1-change-the-state-of-matter-of",
        "2-1": "task-10-use-thermometer",
        "2-2": "task-10-measure-melting-point-(known-substance)",
        "2-3": "task-10-measure-melting-point-(unknown-substance)",
        "3-1": "task-2-power-component",
        "3-2": "task-2-power-component-(renewable-vs-nonrenewable-energy)",
        "3-3": "task-2a-test-conductivity",
        "3-4": "task-2a-test-conductivity-of-unknown-substances",
        "4-1": "task-3-find-living-thing",
        "4-2": "task-3-find-non-living-thing",
        "4-3": "task-3-find-plant",
        "4-4": "task-3-find-animal",
        "5-1": "task-4-grow-plant",
        "5-2": "task-4-grow-fruit",
        "6-1": "task-5-chemistry-mix",
        "6-2": "task-5-chemistry-mix-paint-(secondary-color)",
        "6-3": "task-5-chemistry-mix-paint-(tertiary-color)",
        "7-1": "task-6-lifespan-(longest-lived)",
        "7-2": "task-6-lifespan-(shortest-lived)",
        "7-3": "task-6-lifespan-(longest-lived-then-shortest-lived)",
        "8-1": "task-7-identify-life-stages-1",
        "8-2": "task-7-identify-life-stages-2",
        "9-1": "task-8-inclined-plane-determine-angle",
        "9-2": "task-8-inclined-plane-friction-(named-surfaces)",
        "9-3": "task-8-inclined-plane-friction-(unnamed-surfaces)",
        "10-1": "task-9-mendellian-genetics-(known-plant)",
        "10-2": "task-9-mendellian-genetics-(unknown-plant)"
    }

    env = ScienceWorldEnv()
    for task_id, former_task_name in FORMER_TASK_NAMES.items():
        env.load(task_id)
        obs1, info1 = env.reset()

        # Load a task using its former name.
        env.load(former_task_name)
        obs2, info2 = env.reset()
        assert obs1 == obs2, f"{task_id} is not the same as {former_task_name}"
        assert info1 == info2, f"{task_id} is not the same as {former_task_name}"

        env.server.load(former_task_name, 0, "", False)
        obs2, _, _, info2 = env.step("look around")
        assert obs1 == obs2, f"{task_id} is not the same as {former_task_name}"
        assert info1 == info2, f"{task_id} is not the same as {former_task_name}"

        # Load a task using its new name.
        task_name = env.tasks[task_id]
        env.load(task_name)
        obs2, info2 = env.reset()
        assert obs1 == obs2, f"{task_id} is not the same as {task_name}"
        assert info1 == info2, f"{task_id} is not the same as {task_name}"


def test_consistent_task_names():
    """Verify that public task names come from tasks.json."""
    env = ScienceWorldEnv()
    assert sorted(env.task_names) == sorted(VISIBLE_ID2TASK.values())
    assert sorted(env.get_task_names()) == sorted(VISIBLE_ID2TASK.values())


def test_hidden_tasks_loadable_by_manifest():
    env = ScienceWorldEnv()

    hidden_tasks = [task for task in TASKS if not task.get("visible_in_task_list", True)]
    visible_task_names = set(env.get_task_names())

    for task in hidden_tasks:
        assert task["task_name"] not in visible_task_names
        env.load(task["task_id"])
        assert env.get_max_variations(task["task_name"]) >= 0


def test_obj_tree():
    env = ScienceWorldEnv("1-1")
    env.reset()
    obj_tree = env.getObjectTree()
    print(obj_tree)


def test_final_project_recipe_api():
    corpora = []

    for task_name in [
        "recipe-pipeline-tiny",
        "recipe-pipeline-seen",
        "recipe-pipeline-unseen",
    ]:
        env = CS103ScienceWorldFinalProjectEnv(task_name)
        env.reset()
        corpora.append(env.get_corpus())

    assert len(corpora[0]) == 50
    assert corpora[0] == corpora[1] == corpora[2]

    def extract_product_name(doc):
        header = doc.splitlines()[0]
        for prefix in ("Recipe card for ", "Kitchen note for ", "Lab memo for "):
            if header.startswith(prefix):
                return header[len(prefix):-1]
        return header

    def count_docs_for(product_name):
        return sum(
            extract_product_name(doc) == product_name
            for doc in corpora[2]
        )

    assert count_docs_for("bread") == 3
    assert count_docs_for("jam sandwich") == 3
    assert count_docs_for("peanut butter sandwich") == 3
    assert count_docs_for("banana sandwich") == 3
    assert count_docs_for("peanut butter with jam sandwich") == 3
    assert count_docs_for("peanut butter with banana sandwich") == 3
    assert all(doc.startswith(("Recipe card for ", "Kitchen note for ", "Lab memo for ")) for doc in corpora[0])


def test_final_project_recipe_api_empty_for_non_recipe_tasks():
    recipe_env = CS103ScienceWorldFinalProjectEnv("recipe-pipeline-unseen")
    recipe_env.reset()
    recipe_corpus = recipe_env.get_corpus()

    env = CS103ScienceWorldFinalProjectEnv("corrode-circuit-unseen")
    env.reset()

    assert len(env.get_corpus()) == 50
    assert env.get_corpus() == recipe_corpus


def test_final_project_tasks_have_multiple_variations():
    env = CS103ScienceWorldFinalProjectEnv()

    assert env.get_max_variations("recipe-pipeline-tiny") > 1
    assert env.get_max_variations("recipe-pipeline-seen") > 1
    assert env.get_max_variations("recipe-pipeline-unseen") > 1
    assert env.get_max_variations("corrode-circuit-tiny") > 1
    assert env.get_max_variations("corrode-circuit-seen") > 1
    assert env.get_max_variations("corrode-circuit-unseen") > 1


def test_final_project_corpus_embedding_uses_packaged_numpy_asset(tmp_path, monkeypatch):
    corpus = ["doc-1", "doc-2"]
    expected_embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    embeddings_path = tmp_path / "final_project_corpus_embeddings_bge_m3.npy"
    metadata_path = tmp_path / "final_project_corpus_embeddings_bge_m3.metadata.json"

    np.save(embeddings_path, expected_embeddings)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "BAAI/bge-m3",
                "dtype": "float32",
                "shape": [2, 2],
                "num_docs": 2,
                "embedding_dim": 2,
                "corpus_sha256": scienceworld_module._hash_corpus_documents(corpus),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(scienceworld_module, "FINAL_PROJECT_CORPUS_EMBEDDINGS_PATH", str(embeddings_path))
    monkeypatch.setattr(
        scienceworld_module,
        "FINAL_PROJECT_CORPUS_EMBEDDINGS_METADATA_PATH",
        str(metadata_path),
    )
    scienceworld_module._load_final_project_corpus_embeddings.cache_clear()

    env = object.__new__(CS103ScienceWorldFinalProjectEnv)
    env._gateway = None
    env.get_corpus = lambda: list(corpus)

    loaded_embeddings = env.get_corpus_embedding()

    assert isinstance(loaded_embeddings, np.ndarray)
    assert loaded_embeddings.shape == (2, 2)
    assert loaded_embeddings.dtype == np.float32
    assert np.array_equal(loaded_embeddings, expected_embeddings)

    loaded_embeddings[0, 0] = 99.0
    reloaded_embeddings = env.get_corpus_embedding()
    assert reloaded_embeddings[0, 0] == pytest.approx(1.0)


def test_final_project_corpus_embedding_detects_corpus_hash_mismatch(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "final_project_corpus_embeddings_bge_m3.npy"
    metadata_path = tmp_path / "final_project_corpus_embeddings_bge_m3.metadata.json"

    np.save(embeddings_path, np.asarray([[1.0, 2.0]], dtype=np.float32))
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "BAAI/bge-m3",
                "dtype": "float32",
                "shape": [1, 2],
                "num_docs": 1,
                "embedding_dim": 2,
                "corpus_sha256": "not-the-real-hash",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(scienceworld_module, "FINAL_PROJECT_CORPUS_EMBEDDINGS_PATH", str(embeddings_path))
    monkeypatch.setattr(
        scienceworld_module,
        "FINAL_PROJECT_CORPUS_EMBEDDINGS_METADATA_PATH",
        str(metadata_path),
    )
    scienceworld_module._load_final_project_corpus_embeddings.cache_clear()

    env = object.__new__(CS103ScienceWorldFinalProjectEnv)
    env._gateway = None
    env.get_corpus = lambda: ["doc-1"]

    with pytest.raises(ValueError, match=r"get_corpus\(\) order/content"):
        env.get_corpus_embedding()


def test_final_project_corpus_embedding_missing_asset_has_regeneration_hint(tmp_path, monkeypatch):
    metadata_path = tmp_path / "final_project_corpus_embeddings_bge_m3.metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        scienceworld_module,
        "FINAL_PROJECT_CORPUS_EMBEDDINGS_PATH",
        str(tmp_path / "missing_embeddings.npy"),
    )
    monkeypatch.setattr(
        scienceworld_module,
        "FINAL_PROJECT_CORPUS_EMBEDDINGS_METADATA_PATH",
        str(metadata_path),
    )
    scienceworld_module._load_final_project_corpus_embeddings.cache_clear()

    env = object.__new__(CS103ScienceWorldFinalProjectEnv)
    env._gateway = None
    env.get_corpus = lambda: ["doc-1"]

    with pytest.raises(FileNotFoundError, match="generate_final_project_corpus_embeddings.py"):
        env.get_corpus_embedding()
