import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.metrics.Specificity")
class Specificity(tf.keras.metrics.Metric):
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_negatives = self.add_weight(
            "true_negatives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives", shape=(len(self.thresholds),), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = tf.math.divide_no_nan(
            self.true_negatives,
            tf.math.add(self.true_negatives, self.false_positives),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_negatives, self.false_positives)
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_and_train(
    project_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    min_layers: int = 2,
    max_layers: int = 5,
    min_units: int = 32,
    max_units: int = 512,
    dropout: float = 0.2,
    min_lr: float = 1e-4,
    max_lr: float = 1e-2,
    objective: kt.Objective = kt.Objective("val_accuracy", direction="max"),
    max_trials: int = 2,
    executions_per_trial: int = 2,
    overwrite: bool = True,
    class_weight: dict = None,
    epochs: int = 100,
    batch_size: int = 1024,
    callbacks: list = None,
):
    num_features = x_train.shape[1]
    num_classes = len(y_train.unique())
    if num_classes == 2:
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        output_layer_units = 1
    else:
        loss = "categorical_crossentropy"
        output_activation = "softmax"
        output_layer_units = num_classes

    if class_weight is None:
        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight = dict(enumerate(class_weight))

    def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
        model = tf.keras.Sequential()

        for i in range(hp.Int("num_layers", min_layers, max_layers)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(f"units_{i}", min_units, max_units),
                    activation=hp.Choice(f"activation_{i}", ["relu", "tanh", "swish"]),
                    input_shape=(num_features,),
                )
            )
        if hp.Boolean("dropout"):
            model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(
            tf.keras.layers.Dense(
                units=output_layer_units, activation=output_activation
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float("learning_rate", min_lr, max_lr, sampling="log")
            ),
            loss=loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                Specificity(),
                tf.keras.metrics.AUC(),
            ],
        )
        return model

    # hyper params search
    tuner = kt.RandomSearch(
        build_model,
        directory=f"output/{project_name}/tuner",
        project_name=project_name,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=overwrite,
    )

    tuner.search(
        x_train,
        y_train,
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=f"output/{project_name}/board"),
            tf.keras.callbacks.CSVLogger(f"output/{project_name}/run.csv", append=True),
            *(callbacks or []),
        ],
        validation_data=(x_val, y_val),
    )

    tuner.results_summary()
    # return all models
    return tuner.get_best_models(num_models=max_trials)
