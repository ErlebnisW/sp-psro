import logging
import os
import time
from typing import Dict, Any

import numpy as np
import cloudpickle
import deepdish
from ray.rllib.agents import Trainer
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import PolicyID
from tables.exceptions import HDF5ExtError

from grl.rllib_tools.safe_convert_to_torch_tensor import safe_convert_to_torch_tensor
from grl.utils.common import datetime_str, ensure_dir
from grl.utils.strategy_spec import StrategySpec
from grl.rllib_tools.modified_policies.mixed_class_eval_policy import MixedClassEvalPolicy

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def save_policy_checkpoint(trainer: Trainer,
                           player: int,
                           save_dir: str,
                           policy_id_to_save: PolicyID,
                           checkpoint_name: str,
                           additional_data: Dict[str, Any]):
    date_time = datetime_str()
    checkpoint_name = f"policy_{checkpoint_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights([policy_id_to_save])[policy_id_to_save]
    br_weights = {k.replace(".", "_dot_"): v for k, v in
                  br_weights.items()}  # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    num_save_attempts = 5

    checkpoint_data = {
        "weights": br_weights,
        "player": player,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
    }
    checkpoint_data.update(additional_data)

    for attempt in range(num_save_attempts):
        try:
            deepdish.io.save(path=checkpoint_path, data=checkpoint_data)
            break
        except HDF5ExtError:
            if attempt + 1 == num_save_attempts:
                raise
            time.sleep(1.0)
    return checkpoint_path


def load_pure_strat(policy: Policy, pure_strat_spec: StrategySpec = None, checkpoint_path: str = None, weights_key: str = "weights", must_have_mixed_class_policy_map_id=False):
    if pure_strat_spec is not None and checkpoint_path is not None:
        raise ValueError("Can only pass pure_strat_spec or checkpoint_path but not both")
    if checkpoint_path is None:
        if hasattr(policy, "policy_spec") and pure_strat_spec == policy.policy_spec:
            return
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
        mixed_class_policy_map_id = pure_strat_spec.metadata.get("mixed_class_policy_map_id")
    else:
        pure_strat_checkpoint_path = checkpoint_path
        mixed_class_policy_map_id = None

    weights = None

    try:
        num_load_attempts = 5
        for attempt in range(num_load_attempts):
            try:
                checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
                weights = checkpoint_data[weights_key]
                mixed_class_policy_map_id = checkpoint_data.get("mixed_class_policy_map_id")
                break
            except (HDF5ExtError, KeyError):
                if attempt + 1 == num_load_attempts:
                    raise
                time.sleep(1.0)

    #TODO use correct exception
    except Exception:
        with open(pure_strat_checkpoint_path, "rb") as pickle_file:
            checkpoint_data = cloudpickle.load(pickle_file)
            weights = checkpoint_data[weights_key]

    weights = {key.replace("_dot_", "."): val for key, val in weights.items()}
    for key in weights.keys():
        weights[key] = np.asarray(weights[key]).copy()
        weights[key].setflags(write=1)

    if must_have_mixed_class_policy_map_id and not mixed_class_policy_map_id:
        raise ValueError(f"No mixed_class_policy_map_id found.")

    if must_have_mixed_class_policy_map_id:
        assert isinstance(policy, MixedClassEvalPolicy), type(policy)

    if mixed_class_policy_map_id:
        assert isinstance(policy, MixedClassEvalPolicy), type(policy)
        policy.set_current_policy_id(policy_id=mixed_class_policy_map_id)
        policy.set_weights(weights=weights, is_non_ray_internal_call=True)
    else:
        policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec


def create_get_pure_strat_cached(cache: dict):
    def load_pure_strat_cached(policy: TorchPolicy, pure_strat_spec):

        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]

        if pure_strat_checkpoint_path in cache:
            weights = cache[pure_strat_checkpoint_path]
        else:
            checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
            weights = checkpoint_data["weights"]
            weights = {k.replace("_dot_", "."): v for k, v in weights.items()}

            weights = safe_convert_to_torch_tensor(weights, device=policy.device)
            cache[pure_strat_checkpoint_path] = weights

        policy.model.load_state_dict(weights)
        policy.policy_spec = pure_strat_spec

    return load_pure_strat_cached
