import os
from typing import Sequence

import pandas as pd
import src.utils.toolkit.time_utils as time_utils
import src.utils.visualization.graph_visualization as vis


def get_test_samples_table(
    all_targets,
    all_preds,
    all_targets_str,
    all_preds_str,
    X_test,
    demographic_data_dict,
    order_by_correctness: bool = False,
):
    test_analysis_data = []
    demo_lookup = demographic_data_dict

    for i, filepath in enumerate(X_test):
        analysis_entry = {
            "sample_index": i,
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "true_label": all_targets[i],
            "true_label_str": all_targets_str[i],
            "predicted_label": all_preds[i],
            "predicted_label_str": all_preds_str[i],
            "correct": all_targets[i] == all_preds[i],
        }

        demo_info = demo_lookup.get(filepath)
        if demo_info is not None:
            analysis_entry.update(demo_info)
        else:
            analysis_entry.update({"age": "filepath not found", "sex": "filepath not found"})

        test_analysis_data.append(analysis_entry)

    df = pd.DataFrame(test_analysis_data)
    if order_by_correctness:
        df = df.sort_values(by="correct", ascending=False).reset_index(drop=True)
    return df


def save_tabular_data(output_dir, file_name, pandas_dataframe, logger=None):
    if not isinstance(pandas_dataframe, pd.DataFrame):
        raise TypeError("pandas_dataframe must be a pandas DataFrame")

    time_stamp = time_utils.get_day_month_year_hour_minute()
    csv_file_name = os.path.join(output_dir, f"{file_name}_{time_stamp}.csv")
    pandas_dataframe.to_csv(csv_file_name, index=False)

    msg = f"Pandas dataframe saved at:\n - {csv_file_name}\n"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_distr_of_demo_data(_dataset_manager, tag_list: Sequence[str] = ("age", "sex")):
    pil_imgs_list = []
    X_train_paths = _dataset_manager.X_train_paths
    X_val_paths = _dataset_manager.X_val_paths
    X_test_paths = _dataset_manager.X_test_paths
    demo_dict = _dataset_manager.demographic_data_dict

    for tag in tag_list:
        def get_values_distr_from_demo_dict_w_path(paths):
            return [demo_dict.get(path, {}).get(tag) for path in paths]

        train_values = get_values_distr_from_demo_dict_w_path(X_train_paths)
        val_values = get_values_distr_from_demo_dict_w_path(X_val_paths)
        test_values = get_values_distr_from_demo_dict_w_path(X_test_paths)

        if tag == "sex":
            pil_img = vis.get_category_chart(tag, train_values, val_values, test_values)
        else:
            train_values = [float(v) if v is not None else None for v in train_values]
            val_values = [float(v) if v is not None else None for v in val_values]
            test_values = [float(v) if v is not None else None for v in test_values]
            pil_img = vis.get_distribution_chart(tag, train_values, val_values, test_values)

        pil_imgs_list.append(pil_img)

    return pil_imgs_list


