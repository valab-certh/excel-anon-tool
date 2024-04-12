from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd


def fix_delete_free_text(value: str) -> float | None | str:
    checker = 2
    if isinstance(value, str) and len(value) >= checker:
        if value[0].isalpha() and value[1:].isdigit():
            return float(value[1:])
        if value[0].isalpha() and value[1].isalpha():
            return None
    return value


def add_noise(value: float, epsilon: int) -> float:
    scale = 1 / epsilon
    noise: float = np.random.laplace(loc=0, scale=scale)  # noqa: NPY002
    return (value + noise)


def apply_differential_privacy(  # noqa: C901, PLR0913, PLR0912
    data: pd.DataFrame,
    columns: dict[str, Any],
    epsilon: int,
    max_change: int,
    min_value: int,
    round_value: bool,  # noqa: FBT001
) -> None:
    for column in columns:
        if column in data.columns:
            data[column] = data[column].apply(lambda x: fix_delete_free_text(x))
            data[column] = pd.to_numeric(data[column], errors="coerce")
            if data[column].dtype.kind in "biufc":
                for i, value in enumerate(data[column]):
                    if value is not None and (not np.isnan(value)):
                        noisy_value = add_noise(value, epsilon)
                        if max_change is not None:
                            if round_value:
                                while (
                                    abs(noisy_value - value) >= max_change
                                    or round(noisy_value) == value
                                ):
                                    noisy_value = add_noise(value, epsilon)
                            else:
                                while (
                                    abs(noisy_value - value) >= max_change
                                    or noisy_value == value
                                ):
                                    noisy_value = add_noise(value, epsilon)
                            if (
                                value == 0
                                and max_change == 1
                                and (min_value == 0)
                                and (round_value is True)
                            ):
                                noisy_value = 1
                        if min_value is not None:
                            noisy_value = max(noisy_value, min_value)
                        if round_value:
                            noisy_value = round(noisy_value)
                        data.at[i, column] = noisy_value  # noqa: PD008


def ids_mapping_patient_number(
    patient_number: str,
    data_name: str,
    folder_path: str,
) -> str:
    ids_mapping = {"123": "456"}
    json_file = Path(folder_path) / ("id_mapping_" + data_name + ".json")
    with Path.open(json_file) as f:
        json_data = json.load(f)
    if (
        patient_number == "nan"
        or patient_number == ""
        or (patient_number is None)
        or (patient_number == " ")
    ):
        return patient_number
    prefix, suffix = patient_number.split("-")
    if suffix in json_data:
        suffix = json_data[suffix]
    mapped_prefix = ids_mapping.get(prefix, prefix)
    return f"{mapped_prefix}-{suffix}"


def ids_mapping_provider(provider: str) -> str:
    ids_mapping = {
        "123": "456",
        "1": "456",
        "2": "456",
        "3": "456",
        "4": "456",
        "5": "456",
        "6": "456",
        "7": "456",
        "8": "456",
        "9": "456",
        123: "456",
        1: "456",
        2: "456",
        3: "456",
        4: "456",
        5: "456",
        6: "456",
        7: "456",
        8: "456",
        9: "456",
    }
    mapped_provider = ids_mapping.get(provider, provider)
    return f"{mapped_provider}"


def ethnicity_grouping(df: pd.DataFrame, column_description: str) -> None:
    description_list = list(column_description)
    found_irish = [
        element
        for element in description_list
        if isinstance(element, str) and "Irish" in element
    ]
    flag = False
    if found_irish:
        flag = True
    ethnicity_mapping = {
        1: "White",
        2: "White",
        3: "White",
        4: "African",
        5: "African",
        6: "Asian",
        7: "Asian",
        8: "Asian",
        9: "Arabic",
        10: "Mixed",
        11: "Other",
        "1": "White",
        "2": "White",
        "3": "White",
        "4": "African",
        "5": "African",
        "6": "Asian",
        "7": "Asian",
        "8": "Asian",
        "9": "Arabic",
        "10": "Mixed",
        "11": "Other",
        "Greek": "White",
    }
    if flag:
        ethnicity_mapping = {
            1: "White",
            2: "White",
            3: "White",
            4: "White",
            5: "African",
            6: "African",
            7: "Asian",
            8: "Asian",
            9: "Asian",
            10: "Arabic",
            11: "Mixed",
            12: "Other",
            "1": "White",
            "2": "White",
            "3": "White",
            "4": "White",
            "5": "African",
            "6": "African",
            "7": "Asian",
            "8": "Asian",
            "9": "Asian",
            "10": "Arabic",
            "11": "Mixed",
            "12": "Other",
            "Greek": "White",
        }
    df["Ethnicity"] = df["Ethnicity"].replace(ethnicity_mapping)


def split_string(s: str) -> list[str]:
    pattern = "[,;/]+"
    return [word.strip() for word in re.split(pattern, s) if word.strip()]


def process_string(input_str: str) -> str:
    pattern = "^[A-Za-z]\\d\\d[A-Za-z]{2}(\\d{1,2})?$"
    star_space_flag = False
    end_space_flag = False
    input_str = input_str.strip()
    substrings = split_string(input_str)
    processed_substrings = []
    for sub in substrings:
        modified_sub = sub
        if sub.startswith(" "):
            star_space_flag = True
            modified_sub = sub.lstrip()
        if sub.endswith(" "):
            end_space_flag = True
            modified_sub = modified_sub.rstrip()
        if bool(re.match(pattern, modified_sub)):
            level_up_substring = modified_sub
            if len(modified_sub) == 7:  # noqa: PLR2004
                level_up_substring = modified_sub[:-2]
                if star_space_flag:
                    level_up_substring = " " + level_up_substring
                if end_space_flag:
                    level_up_substring = level_up_substring + " "
            processed_substrings.append(level_up_substring)
    return ",".join(processed_substrings)


def medication_levelup(cancer_type: str, df: pd.DataFrame) -> None:  # noqa: C901
    if cancer_type == "breast":
        columns_to_check = [
            "Medications",
            "Type of CTX",
            "Type of CIT",
            "Type of CRT",
            "Type of RT",
            "Type of HT",
            "Type of TT",
            "Type of IT",
        ]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)
    if cancer_type == "colorectal":
        columns_to_check = ["Type of CT", "Type of CRT", "Type of CIT"]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)
    if cancer_type == "lung":
        columns_to_check = [
            "Type of CT",
            "Type of CRT",
            "Type of CIT",
            "Type of TT",
            "Type of IT",
        ]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)
    if cancer_type == "prostate":
        columns_to_check = ["Medications"]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)


def remove_dot_and_digit(input_string: str | list[str]) -> str | list[str]:
    if isinstance(input_string, str):
        input_string = input_string.strip()
        substrings = input_string.split(",")
        pattern = r"[A-Za-z]\d{2}(\.\d+)?$"
        processed_substrings = [
            substring for substring in substrings if re.match(pattern, substring)
        ]
        processed_substrings = [
            re.sub(r"\.\d", "", substring) for substring in processed_substrings
        ]
        return ",".join(processed_substrings)

    processed_strings = []
    for item in input_string:
        string = item.strip()
        pattern = r"[A-Za-z]\d{2}(\.\d+)?$"
        if re.match(pattern, string):
            processed_string = re.sub(r"\.\d", "", string)
            processed_strings.append(processed_string)
    return processed_strings


def calculate_jaccard_similarity(  # noqa: C901, PLR0912
    df_original: pd.DataFrame,
    df_anonymized: pd.DataFrame,
    data_name: str,
    sheat: str,
) -> float | None:
    qi = []
    if sheat == "General info":
        if "Ethnicity" in df_original.columns:
            if data_name == "breast":
                qi = ["Medical History", "Ethnicity", "Medications"]
            if data_name == "colorectal":
                qi = ["Medical History", "Ethnicity"]
            if data_name == "lung":
                qi = ["Medical History", "Ethnicity"]
            if data_name == "prostate":
                qi = ["Medical History", "Ethnicity", "Medications"]
        else:
            if data_name == "breast":
                qi = ["Medical History", "Medications"]
            if data_name == "colorectal":
                qi = ["Medical History"]
            if data_name == "lung":
                qi = ["Medical History"]
            if data_name == "prostate":
                qi = ["Medical History", "Medications"]
    if sheat == "Treatment":
        if data_name == "breast":
            qi = [
                "Type of CTX",
                "Type of CIT",
                "Type of CRT",
                "Type of RT",
                "Type of HT",
                "Type of TT",
                "Type of IT",
            ]
        if data_name == "colorectal":
            qi = ["Type of CT", "Type of CRT", "Type of CIT"]
        if data_name == "lung":
            qi = [
                "Type of CT",
                "Type of CRT",
                "Type of CIT",
                "Type of TT",
                "Type of IT",
            ]
    if qi:
        df_anonymized = df_anonymized.replace("nan", np.nan)
        categories1 = set(df_original[qi].astype(str).values.flatten())  # noqa: PD011
        categories2 = set(df_anonymized[qi].astype(str).values.flatten())  # noqa: PD011
        intersection = len(categories1.intersection(categories2))
        union = len(categories1.union(categories2))
        return intersection / union if union != 0 else 0
    return None


def calculate_generalization_level(
    original_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
) -> None:
    columns = [
        "Medical History",
        "Ethnicity",
        "Medications",
        "Type of CT",
        "Type of CTX",
        "Type of CIT",
        "Type of CRT",
        "Type of RT",
        "Type of HT",
        "Type of TT",
        "Type of IT",
    ]
    for column_name in columns:
        if column_name in original_df.columns:
            original_df[column_name] = original_df[column_name].fillna("").astype(str)
            anonymized_df[column_name] = (
                anonymized_df[column_name].fillna("").astype(str)
            )
            values_or = {
                item.strip()
                for sublist in original_df[column_name].str.split(",")
                for item in sublist
            }
            values_an = {
                item.strip()
                for sublist in anonymized_df[column_name].str.split(",")
                for item in sublist
            }
            common_values = values_or.intersection(values_an)
            exclusive_values_original = values_or - values_an
            exclusive_values_anonymized = values_an - values_or
            total_unique_values_original = len(values_or)
            total_unique_values_anonymized = len(values_an)
            common_values_count = len(common_values)
            len(exclusive_values_original)
            exclusive_anonymized_count = len(exclusive_values_anonymized)
            common_values_count / total_unique_values_anonymized * 100 if total_unique_values_anonymized else 0  # noqa: E501
            common_values_count / total_unique_values_original * 100 if total_unique_values_original else 0  # noqa: E501
            exclusive_anonymized_count / total_unique_values_anonymized * 100 if total_unique_values_anonymized else 0  # noqa: E501


def read_differential_arguments(
    differential_pr_arguments: Collection[str],
) -> dict[str, Any]:
    columns = list(differential_pr_arguments)
    return {"columns": columns}


def get_cancer_type(value: str) -> str:
    if value is not None:
        lowercase_value = value.lower()
        cancer_types = ["breast", "colorectal", "lung", "prostate"]
        for cancer_type in cancer_types:
            if cancer_type.lower() in lowercase_value:
                return cancer_type
    return ""


def skip_rows_array(value: str) -> list[int]:
    length = len(value)
    result = list(range(length + 1))
    result.append(length + 2)
    return result


def find_patien_number(
    header_row: list[str],
    df: pd.DataFrame,
) -> tuple[bool, str, str]:
    flag = False
    column_discription = ""
    unwanted_rows = ""
    if "Patient Number*" in header_row or "Patient Number" in header_row:
        column_discription = df.iloc[0].values  # noqa: PD011
    elif (
        "Patient Number*" in df.iloc[:1].values  # noqa: PD011
        or "Patient Number" in df.iloc[:1].values  # noqa: PD011
    ):
        flag = True
        column_discription = df.iloc[1].values  # noqa: PD011
    else:
        for i in range(10):
            row = df.iloc[:i].values  # noqa: PD011
            if "Patient Number*" in row or "Patient Number" in row:
                unwanted_rows = df.iloc[: i - 1].values  # noqa: PD011
                column_discription = df.iloc[i].values  # noqa: PD011
                break
    return (flag, column_discription, unwanted_rows)


class Anonymizer:
    def __init__(  # noqa: C901, PLR0913, PLR0912, PLR0915
        self: Anonymizer,
        excel_path: Path,
        sheat_name: str,
        data_name: str,
        sheat: str,
        folder_path: str,
        data_provider: str,
    ) -> None:
        self.sheat_name = sheat_name
        self.excelName = data_name
        self.data_name = get_cancer_type(data_name)
        self.fileName = self.excelName + "-" + self.sheat_name + "-anonymized.xlsx"
        self.data_provider = data_provider
        xlxs_path = excel_path
        datframe = pd.read_excel(xlxs_path, sheet_name=sheat)
        column_indices_with_value = [
            i
            for i, column in enumerate(datframe.columns)
            if datframe[column].eq("Year of birth").any()
        ]
        column_name_to_drop = datframe.columns[column_indices_with_value]
        datframe = datframe.drop(column_name_to_drop, axis=1)
        self.header_row = datframe.columns.tolist()
        self.flag, self.column_discription, self.unwanted_rows = find_patien_number(
            self.header_row,
            datframe,
        )
        if len(self.unwanted_rows) == 0 and self.flag is False:
            skip_rows = [1]
        elif self.flag is True:
            skip_rows = [0, 2]
        else:
            skip_rows = skip_rows_array(self.unwanted_rows)
        del datframe
        datframe = pd.read_excel(xlxs_path, skiprows=skip_rows, sheet_name=sheat)
        if sheat == "General info" and "Year of birth" in datframe.columns:
            datframe = datframe.drop("Year of birth", axis=1)
        differential_pr_arguments = {
            "breast": {
                "Baseline": {},
                "General_info": {"Age at diagnosis", "Delivery Time"},
                "Histology_Mutations": {
                    "Date of Biopsy*",
                    "Date of Biopsy",
                    "Date",
                    "Date*",
                },
                "Lab_Results": {"Date*Date"},
                "Timepoints": {" Date*", "Date", "Delivery Time"},
                "Treatment": {
                    "Date of Surgery",
                    "Date of CTX",
                    "Date of CIT",
                    "Date of CRT",
                    "Date of RT",
                    "Dateof HT",
                    "Date of TT",
                    "Date of IT",
                },
            },
            "colorectal": {
                "Baseline": {},
                "General_info": {"Age at diagnosis", "Delivery Time"},
                "Histology_Mutations": {"Biopsy Date*", "Surgery Date"},
                "Lab_Results": {"Date*Date"},
                "Timepoints": {"Date*", "Date", "Delivery Time"},
                "Treatment": {
                    "Date of surgery",
                    "Date of last CT",
                    "Date of last CRT",
                    "Date of last CIT",
                    "Date of last RT",
                    "Date of post-treatment surgery",
                },
            },
            "lung": {
                "Baseline": {},
                "General_info": {
                    "Age at diagnosisyears of smoking",
                    "# of cigarettes per day",
                    "age quitting smoking",
                    "Delivery Time",
                },
                "Histology_Mutations": {"Biopsy Date*", "Surgery Date"},
                "Lab_Results": {"Date*Date"},
                "Timepoints": {"Date*", "Date", "Delivery Time"},
                "Treatment": {
                    "Surgery Date",
                    "Date of last CT",
                    "Date of last CRT",
                    "Date of last CIT",
                    "Date of last TT",
                    "Date of last IT",
                    "Date of last RT",
                    "Date of post-treatment surgery",
                },
            },
            "prostate": {
                "Baseline": {},
                "General_info": {"Age at diagnosis", "Delivery Time"},
                "Histology_Mutations": {},
                "Lab_Results": {"Date*Date"},
                "Timepoints": {
                    "Date",
                    "Date - PSA 1",
                    "Date - PSA 2",
                    "Date - PSA 3Date - PSA 4",
                    "Date - PSA 5",
                    "Date of biochemical recurrence",
                },
                "Treatment": {
                    "Date of surgery",
                    "Date of last CT",
                    "Date of last CRT",
                    "Date of last CIT",
                    "Date of last RT",
                    "Date of post-treatment surgery",
                },
            },
        }
        differential_pr_params = read_differential_arguments(
            differential_pr_arguments[self.data_name][sheat_name],
        )
        if differential_pr_params is not None:
            columns_list = differential_pr_params["columns"]
            apply_differential_privacy(datframe, columns_list, 1, 1, 0, True)  # noqa: FBT003
        df_original = datframe.copy()
        medication_levelup(self.data_name, datframe)
        if "Medical History" in datframe.columns:
            datframe["Medical History"] = datframe["Medical History"].fillna("")
            datframe["Medical History"] = datframe["Medical History"].apply(
                remove_dot_and_digit,
            )
            datframe["Medical History"] = datframe["Medical History"].replace(
                "",
                np.nan,
            )
        if sheat == "General info" and "Ethnicity" in datframe.columns:
            ethnicity_grouping(datframe, self.column_discription)
        if "Patient Number*" in datframe.columns:
            datframe["Patient Number*"] = datframe["Patient Number*"].astype(str)
            datframe["Patient Number*"] = datframe["Patient Number*"].apply(
                ids_mapping_patient_number,
                data_name=self.data_name,
                folder_path=folder_path,
            )
        if "Patient Number" in datframe.columns:
            datframe["Patient Number"] = datframe["Patient Number"].astype(str)
            datframe["Patient Number"] = datframe["Patient Number"].apply(
                ids_mapping_patient_number,
                data_name=self.data_name,
                folder_path=folder_path,
            )
        if sheat == "General info" and "Provider*" in datframe.columns:
            datframe["Provider*"] = datframe["Provider*"].apply(ids_mapping_provider)
        if sheat == "General info" and "Provider" in datframe.columns:
            datframe["Provider"] = datframe["Provider"].apply(ids_mapping_provider)
        df_anonymized = datframe.copy()
        jaccard_similarity = calculate_jaccard_similarity(
            df_original,
            df_anonymized,
            self.data_name,
            sheat,
        )
        if jaccard_similarity:
            pass
        calculate_generalization_level(df_original, df_anonymized)
        if self.flag is True:
            headers = datframe.columns.tolist()
            datframe.columns = self.header_row
            datframe = pd.concat(
                [
                    datframe.iloc[:0],
                    pd.DataFrame([headers], columns=datframe.columns),
                    datframe.iloc[0:],
                ],
            )
            datframe = pd.concat(
                [
                    datframe.iloc[:1],
                    pd.DataFrame([self.column_discription], columns=datframe.columns),
                    datframe.iloc[1:],
                ],
            )
        if len(self.unwanted_rows) == 0 and self.flag is False:
            datframe = pd.concat(
                [
                    datframe.iloc[:0],
                    pd.DataFrame([self.column_discription], columns=datframe.columns),
                    datframe.iloc[0:],
                ],
            )
        if len(self.unwanted_rows) != 0:
            headers = datframe.columns.tolist()
            datframe.columns = self.header_row
            datframe = pd.concat(
                [
                    datframe.iloc[:0],
                    pd.DataFrame([headers], columns=datframe.columns),
                    datframe.iloc[0:],
                ],
            )
            for i in range(len(self.unwanted_rows)):
                datframe = pd.concat(
                    [
                        datframe.iloc[:i],
                        pd.DataFrame([self.unwanted_rows[i]], columns=datframe.columns),
                        datframe.iloc[i:],
                    ],
                )
            datframe = pd.concat(
                [
                    datframe.iloc[: len(self.unwanted_rows) + 1],
                    pd.DataFrame([self.column_discription], columns=datframe.columns),
                    datframe.iloc[len(self.unwanted_rows) + 1 :],
                ],
            )

        pre_res_folder = Path("results") / self.data_name
        res_folder = Path(pre_res_folder) / self.data_provider
        self.anon_folder = res_folder
        self.fold = folder_path
        self.resultFilename = Path(self.anon_folder) / self.fileName
        Path(self.anon_folder).mkdir(parents=True, exist_ok=True)
        datframe.to_excel(self.resultFilename, sheet_name=sheat, index=False)

    def xlsx_to_excel(self: Anonymizer, excel_sheets: dict[str, str]) -> None:
        output_path = Path(self.fold) / (self.excelName + "_anonymized" + ".xls")
        excel_writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
        for xlsx_file in os.listdir(self.anon_folder):
            if "anonymized" in xlsx_file:
                file_name = xlsx_file.split("-")[1]
                replacement_key = next(
                    (key for key, value in excel_sheets.items() if value == file_name),
                    None,
                )
                if replacement_key:
                    file_name = replacement_key
                if xlsx_file.endswith(".xlsx"):
                    dataframe = pd.read_excel(Path(self.anon_folder) / xlsx_file)
                    dataframe.to_excel(excel_writer, sheet_name=file_name, index=False)
        excel_writer._save()  # noqa: SLF001

    def delete_files(self: Anonymizer, names_to_delete: set[str]) -> None:
        files = os.listdir(self.anon_folder)
        for file_name in files:
            if file_name.endswith(".xlsx") and any(
                name in file_name for name in names_to_delete
            ):
                file_path = Path(self.anon_folder) / file_name
                Path(file_path).unlink()
        shutil.rmtree("results/")


def exec_anonymization(excel_path: Path, folder_path: str) -> None:
    data_provider = excel_path.parts[-3]
    excel_sheats = {
        "General info": "General_info",
        "Timepoints": "Timepoints",
        "Baseline": "Baseline",
        "Histology - Mutations": "Histology_Mutations",
        "Treatment": "Treatment",
        "Lab Results": "Lab_Results",
    }
    names_to_delete = {
        "General_info",
        "Timepoints",
        "Baseline",
        "Histology_Mutations",
        "Treatment",
        "Lab_Results",
    }
    sheats_names = pd.ExcelFile(excel_path).sheet_names
    excel_file_name = Path(excel_path).name.split(".")[0]
    for sheat_name in sheats_names:
        sheat = excel_sheats[sheat_name]
        anonymizer = Anonymizer(
            excel_path=excel_path,
            sheat_name=sheat,
            data_name=excel_file_name,
            sheat=sheat_name,
            folder_path=folder_path,
            data_provider=data_provider,
        )
    anonymizer.xlsx_to_excel(excel_sheets=excel_sheats)
    anonymizer.delete_files(names_to_delete=names_to_delete)


def anonymize_excel(folder_path: str) -> None:
    cancer_types = ["breast", "colorectal", "lung", "prostate"]
    excel_suffixes = [
        "_cancer.xls",
        "_cancer_training.xls",
        "_cancer_observational.xls",
        "_cancer_feasibility.xls",
        "_cancer.xlsx",
        "_cancer_training.xlsx",
        "_cancer_observational.xlsx",
        "_cancer_feasibility.xlsx",
    ]
    possible_file_names = [
        f"{cancer_type}{suffix}".lower()
        for cancer_type in cancer_types
        for suffix in excel_suffixes
    ]
    for filename in os.listdir(folder_path):
        if filename.lower() in possible_file_names:
            file_path = Path(folder_path) / filename
            exec_anonymization(file_path, folder_path)


def main_cli() -> None:
    import fire

    fire.Fire(anonymize_excel)


if __name__ == "__main__":
    anonymize_excel("prm/samples/valab/data")
