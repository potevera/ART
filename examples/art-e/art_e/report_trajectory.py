from art_e.rollout import ProjectTrajectory
import weave
from weave.trace.autopatch import AutopatchSettings

import art

# weave.init is expensive; reuse one client per project across trajectories.
_weave_clients: dict = {}


def _get_weave_client(project: str):
    if project not in _weave_clients:
        _weave_clients[project] = weave.init(
            project, autopatch_settings=AutopatchSettings(disable_autopatch=True)
        )
    return _weave_clients[project]


def report_trajectory(
    model: art.Model,
    trajectory: ProjectTrajectory,
    step: int = 0,
):
    client = _get_weave_client(model.project)

    inputs = {
        "model": model.name,
        "scenario": trajectory.scenario,
        "step": step,
    }

    if isinstance(model, art.TrainableModel):
        inputs["base_model"] = model.base_model

    call = client.create_call(
        "trajectory",
        inputs=inputs,
    )
    client.finish_call(call, output={"tr": trajectory})
