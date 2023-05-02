import pandas as pd
from celery import Task
from celery.utils.log import get_task_logger

from frontend.app.blueprints.util import choose_model, estimate_n_clusters
from frontend.app.cache import cache
from frontend.app.celery_app import celery_app
from frontend.app.model import Dataset
from frontend.subgroup_detection.fairness import test_model_fairness

log = get_task_logger(__name__)


class FairnessTask(Task):
    """
    Utility class for user-bound fairness tasks (one at a time).
    Automatically caches and uncaches the task. Also revokes
    an already running task for this user if started again.
    """

    @staticmethod
    def _cache_key(current_user):
        return "fair_task_" + current_user.session_token

    @staticmethod
    def cache(current_user, task_id):
        cache.add(FairnessTask._cache_key(current_user), task_id, timeout=0)  # no expiration

    @staticmethod
    def get(current_user):
        return cache.get(FairnessTask._cache_key(current_user))

    @staticmethod
    def delete(current_user):
        return cache.delete(FairnessTask._cache_key(current_user))

    @staticmethod
    def stop(current_user):
        task_id = FairnessTask.get(current_user)

        # If there is already a task running, stop it
        if task_id is not None:
            # Delete cache entry
            FairnessTask.delete(current_user)

            # Revoke task (terminates if running already)
            fairness_analysis.AsyncResult(task_id).revoke(terminate=True)
            log.debug(f"Revoked task with id {task_id}")
            return True
        return False

    def on_success(self, retval, task_id, args, kwargs):
        log.info("On success")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        log.info("On failure")
        # TODO task failure


@celery_app.task(bind=True, base=FairnessTask)
def fairness_analysis(self, df_json, algorithm, pos_label=1, threshold=0.65, categ_columns=None,
                      label_column='class', prediction_column='out', param_dict=None, estimate_k=False):
    log.info(f"Starting fairness analysis: algorithm={algorithm}, pos_label={pos_label}, "
             f"threshold={threshold}, categ_columns={categ_columns}")

    def progress(status):
        self.update_state(state='PROGRESS', meta={'status': status})

    # Load data
    progress('Loading data ...')
    data = pd.read_json(df_json)  # deserialize json

    # If estimate_k is True, apply xmeans to get an estimate of the number of clusters k
    if estimate_k:          # todo: this might be wrong for "automatic" clustering methods like DBSCAN
        k = estimate_n_clusters(data, categ_columns, label_column, prediction_column)
        log.info(f"Estimated n clusters: {k}")
        param_dict['n_clusters'] = k  # overwrites previous setting

    # Specify model
    model = choose_model(algorithm, param_dict)
    log.info(f"Model {model}")

    # Test fairness of the classification model
    fair_res = test_model_fairness(data, model=model, pos_label=pos_label, threshold=threshold,
                                   categ_columns=categ_columns, progress=progress, label_column=label_column,
                                   prediction_column=prediction_column)

    # Return result as json
    return fair_res.to_json()
