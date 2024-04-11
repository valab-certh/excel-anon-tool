import json
import os
import re
import shutil

import numpy as np
import pandas as pd


def fix_delete_free_text(value: str) -> float:
    # Check if the value is a string and has at least two characters
    checker = 2
    if isinstance(value, str) and len(value) >= checker:
        if value[0].isalpha() and value[1:].isdigit():
            # Remove the first character
            return float(value[1:])
        if value[0].isalpha() and value[1].isalpha():
            return None
    return value


def add_noise(value, epsilon):
    # Laplace noise generation
    scale = 1 / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return value + noise


def apply_differential_privacy(
    data: any,
    columns: list,
    epsilon: int,
    max_change: int,
    min_value: int,
    round_value: bool,
) -> None:
    
    for column in columns:
        if column in data.columns:
            data[column] = data[column].apply(lambda x: fix_delete_free_text(x))

            data[column] = pd.to_numeric(data[column], errors="coerce")

            if data[column].dtype.kind in "biufc":
                # Set sensitivity to max_change to limit the change to max ±max_change

                for i, value in enumerate(data[column]):
                    if value is not None and not np.isnan(value):
                        noisy_value = add_noise(
                            value,
                            epsilon,
                        )

                        # Apply max_change if specified
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
                                and min_value == 0
                                and round_value is True
                            ):
                                noisy_value = 1

                        # Apply min_value if specified
                        if min_value is not None:
                            noisy_value = max(noisy_value, min_value)

                        if round_value:
                            noisy_value = round(noisy_value)

                        data.at[i, column] = noisy_value


def ids_mapping_patient_number(patient_number, data_name, folder_path):
    ids_mapping = {
        "123": "456",
    }

    json_file = os.path.join(folder_path, "id_mapping_" + data_name + ".json")
    with open(json_file) as f:
        json_data = json.load(f)

    if (
        patient_number == np.nan
        or patient_number == "nan"
        or patient_number == ""
        or (patient_number is None)
        or patient_number == " "
    ):
        return patient_number

    prefix, suffix = patient_number.split("-")
    if suffix in json_data:
        suffix = json_data[suffix]
    mapped_prefix = ids_mapping.get(
        prefix,
        prefix,
    )  # Default to original if not found in mapping
    return f"{mapped_prefix}-{suffix}"


def ids_mapping_provider(provider) -> str:
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
    mapped_provider = ids_mapping.get(
        provider,
        provider,
    )  # Default to original if not found in mapping
    return f"{mapped_provider}"


def ethnicityGrouping(df, columnDescription) -> None:
    descriptionList = list(columnDescription)
    found_irish = [
        element
        for element in descriptionList
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

    # Step 3: Replace values in the 'Ethnicity' column using the mapping dictionary
    df["Ethnicity"] = df["Ethnicity"].replace(ethnicity_mapping)


def split_string(s):
    # Define a pattern that includes various delimiters: commas, semicolons, slashes, spaces, etc.
    pattern = r"[,;/]+"
    # Split the string based on the pattern and filter out any empty strings
    return [word.strip() for word in re.split(pattern, s) if word.strip()]


def process_string(input_str):
    pattern = r"^[A-Za-z]\d\d[A-Za-z]{2}(\d{1,2})?$"
    star_space_flag = False
    end_space_flag = False
    input_str = input_str.strip()
    substrings = split_string(input_str)
    processed_substrings = []

    for substring in substrings:
        if substring.startswith(" "):
            star_space_flag = True
            substring = substring.lstrip()
        if substring.endswith(" "):
            end_space_flag = True
            substring = substring.rstrip()

        if bool(re.match(pattern, substring)):
            level_up_substring = substring
            if len(substring) == 7:
                level_up_substring = substring[:-2]
                if star_space_flag:
                    level_up_substring = " " + level_up_substring
                if end_space_flag:
                    level_up_substring = level_up_substring + " "

            processed_substrings.append(level_up_substring)

    return ",".join(processed_substrings)


def medicationLevelUp(cancerType, df) -> None:
    if cancerType == "breast":
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

    if cancerType == "colorectal":
        columns_to_check = ["Type of CT", "Type of CRT", "Type of CIT"]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)

    if cancerType == "lung":
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

    if cancerType == "prostate":
        columns_to_check = ["Medications"]
        for column in columns_to_check:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(process_string)


def all_values_match(s, pattern):
    values = re.split(r",|;|/", s)
    return all(re.match(pattern, value) for value in values)


def remove_dot_and_digit(input_string):
    input_string = str(input_string)
    processed_substrings = []
    input_string = input_string.strip()
    substrings = split_string(input_string)
    pattern = r"[A-Za-z]\d{2}(\.\d)?$"

    for substring in substrings:
        if re.match(pattern, substring):
            processed_substrings.append(substring)

    substrings = ",".join(processed_substrings)

    if type(substrings) == int or substrings is None or substrings == "":
        return substrings
    return re.sub(r"\.\d", "", substrings)


def calculate_jaccard_similarity(dfOriginal, dfAnonymized, data_name, sheat):
    qi = []
    if sheat == "General info":
        if "Ethnicity" in dfOriginal.columns:
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
        dfAnonymized = dfAnonymized.replace("nan", np.nan)
        categories1 = set(dfOriginal[qi].astype(str).values.flatten())
        categories2 = set(dfAnonymized[qi].astype(str).values.flatten())

        intersection = len(categories1.intersection(categories2))
        union = len(categories1.union(categories2))

        return intersection / union if union != 0 else 0

    return None


def calculate_generalization_level(original_df, anonymized_df) -> None:
    # column_name = "Medications"
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

            valuesOr = {
                item.strip()
                for sublist in original_df[column_name].str.split(",")
                for item in sublist
            }
            valuesAn = {
                item.strip()
                for sublist in anonymized_df[column_name].str.split(",")
                for item in sublist
            }

            # Calculating metrics
            common_values = valuesOr.intersection(valuesAn)
            exclusive_values_original = valuesOr - valuesAn
            exclusive_values_anonymized = valuesAn - valuesOr

            total_unique_values_original = len(valuesOr)
            total_unique_values_anonymized = len(valuesAn)
            common_values_count = len(common_values)
            len(exclusive_values_original)
            exclusive_anonymized_count = len(exclusive_values_anonymized)

            (
                (common_values_count / total_unique_values_anonymized) * 100
                if total_unique_values_anonymized
                else 0
            )
            (
                (common_values_count / total_unique_values_original) * 100
                if total_unique_values_original
                else 0
            )
            (
                (exclusive_anonymized_count / total_unique_values_anonymized) * 100
                if total_unique_values_anonymized
                else 0
            )


def read_differentialArguments(differential_pr_arguments):
    columns = []

    for c in differential_pr_arguments:
        columns.append(c)

    return {
        "columns": columns,
    }


def convert_columns_to_indexes(data, columnsNames):
    if columnsNames is None:
        return None

    return [data.columns.get_loc(col) for col in columnsNames]


def extract_columns_from_hierarchies(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the extracted strings
    columnsNames = []

    # Flag to check if there are CSV files in the folder
    csv_files_exist = False

    # Iterate through each file
    for file_name in files:
        # Check if the file is a CSV file
        if file_name.endswith(".csv"):
            csv_files_exist = True
            # Split the file name by "_"
            parts = file_name.split("_")

            # If there is more than one part after splitting, take the last part
            if len(parts) > 1:
                last_part = parts[-1]

                # Remove the ".csv" extension and add the extracted string to the list
                columnsNames.append(last_part[:-4])

    # If no CSV files were found, return None
    if not csv_files_exist:
        return None

    return columnsNames


def find_csv_file(folder_path, file_names):
    """
    Search for CSV files in a folder and return the names of files that match the given list of names.

    Parameters:
    - folder_path (str): The path to the folder containing CSV files.
    - file_names (list): A list of strings representing the names to search for.

    Returns:
    - list: A list of CSV file names that were found in the folder.
    """

    # Get a list of files in the folder
    files_in_folder = os.listdir(folder_path)

    # Filter files that are CSV files and match the given names
    return [
        file_name
        for file_name in files_in_folder
        if file_name.endswith(".csv") and file_name in file_names
    ]


def get_cancer_type(value):
    cancer_types = ["breast", "colorectal", "lung", "prostate"]

    # Convert the input value to lowercase for case-insensitive comparison
    lowercase_value = value.lower()

    # Check if any cancer type is present in the lowercase version of the value
    for cancer_type in cancer_types:
        if cancer_type.lower() in lowercase_value:
            return cancer_type

    # If no match is found, return None or any other value you prefer
    return None


def skip_rows_array(value):
    length = len(value)
    result = list(range(length + 1))

    result.append(length + 2)

    return result


def findPatienNumber(header_row, df):
    flag = False
    columnDiscription = ""
    unwantedRows = ""

    if "Patient Number*" in header_row or "Patient Number" in header_row:
        columnDiscription = df.iloc[0].values
    elif (
        "Patient Number*" in df.iloc[:1].values
        or "Patient Number" in df.iloc[:1].values
    ):
        flag = True
        columnDiscription = df.iloc[1].values
    else:
        for i in range(10):
            row = df.iloc[:i].values
            if "Patient Number*" in row or "Patient Number" in row:
                unwantedRows = df.iloc[: i - 1].values
                columnDiscription = df.iloc[i].values
                break
    return flag, columnDiscription, unwantedRows


class Anonymizer:
    def __init__(
        self,
        excel_path=None,
        sheat_Name=None,
        data_name=None,
        sheat=None,
        folder_path=None,
        data_provider=None,
    ) -> None:
        self.sheat_Name = sheat_Name
        self.excelName = data_name
        self.data_name = get_cancer_type(data_name)
        self.fileName = self.excelName + "-" + self.sheat_Name + "-anonymized.xlsx"
        self.dataProvider = data_provider

        # Dataset path
        xlxs_Path = excel_path
        # Data path
        self.path = os.path.join("prm/data", self.data_name)  # trailing /

        # keep the header and the second row
        df = pd.read_excel(xlxs_Path, sheet_name=sheat)

        column_indices_with_value = [
            i
            for i, column in enumerate(df.columns)
            if df[column].eq("Year of birth").any()
        ]

        column_name_to_drop = df.columns[column_indices_with_value]
        df = df.drop(column_name_to_drop, axis=1)

        self.header_row = df.columns.tolist()

        self.flag, self.columnDiscription, self.unwantedRows = findPatienNumber(
            self.header_row,
            df,
        )

        if len(self.unwantedRows) == 0 and self.flag is False:
            skipRows = [1]
        elif self.flag is True:
            skipRows = [0, 2]
        else:
            skipRows = skip_rows_array(self.unwantedRows)

        del df
        df = pd.read_excel(xlxs_Path, skiprows=skipRows, sheet_name=sheat)

        if sheat == "General info" and "Year of birth" in df.columns:
            df = df.drop("Year of birth", axis=1)

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
                "Timepoints": {
                    " Date*",
                    "Date",
                    "Delivery Time",
                },
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
                "Histology_Mutations": {
                    "Biopsy Date*",
                    "Surgery Date",
                },
                "Lab_Results": {"Date*Date"},
                "Timepoints": {
                    "Date*",
                    "Date",
                    "Delivery Time",
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
            "lung": {
                "Baseline": {},
                "General_info": {
                    "Age at diagnosisyears of smoking",
                    "# of cigarettes per day",
                    "age quitting smoking",
                    "Delivery Time",
                },
                "Histology_Mutations": {
                    "Biopsy Date*",
                    "Surgery Date",
                },
                "Lab_Results": {"Date*Date"},
                "Timepoints": {
                    "Date*",
                    "Date",
                    "Delivery Time",
                },
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

        differentialPrParams = read_differentialArguments(
            differential_pr_arguments[self.data_name][sheat_Name],
        )
        if differentialPrParams is not None:
            columns_list = differentialPrParams["columns"]
            apply_differential_privacy(
                df,
                columns_list,
                1,
                1,
                0,
                True,
            )

        dfOriginal = df.copy()
        medicationLevelUp(self.data_name, df)

        if "Medical History" in df.columns:
            df["Medical History"] = df["Medical History"].fillna("")

            df["Medical History"] = df["Medical History"].apply(remove_dot_and_digit)
            df["Medical History"] = df["Medical History"].replace("", np.nan)

        if (sheat == "General info") and "Ethnicity" in df.columns:
            ethnicityGrouping(df, self.columnDiscription)

        if "Patient Number*" in df.columns:
            df["Patient Number*"] = df["Patient Number*"].astype(str)
            df["Patient Number*"] = df["Patient Number*"].apply(
                ids_mapping_patient_number,
                data_name=self.data_name,
                folder_path=folder_path,
            )

        if "Patient Number" in df.columns:
            df["Patient Number"] = df["Patient Number"].astype(str)
            df["Patient Number"] = df["Patient Number"].apply(
                ids_mapping_patient_number,
                data_name=self.data_name,
                folder_path=folder_path,
            )

        if (sheat == "General info") and "Provider*" in df.columns:
            df["Provider*"] = df["Provider*"].apply(ids_mapping_provider)

        if (sheat == "General info") and "Provider" in df.columns:
            df["Provider"] = df["Provider"].apply(ids_mapping_provider)

        dfAnonymized = df.copy()
        jaccard_similarity = calculate_jaccard_similarity(
            dfOriginal,
            dfAnonymized,
            self.data_name,
            sheat,
        )
        if jaccard_similarity:
            pass

        calculate_generalization_level(dfOriginal, dfAnonymized)

        if self.flag is True:
            headers = df.columns.tolist()
            df.columns = self.header_row
            df = pd.concat(
                [df.iloc[:0], pd.DataFrame([headers], columns=df.columns), df.iloc[0:]],
            )
            df = pd.concat(
                [
                    df.iloc[:1],
                    pd.DataFrame([self.columnDiscription], columns=df.columns),
                    df.iloc[1:],
                ],
            )

        if len(self.unwantedRows) == 0 and self.flag is False:
            df = pd.concat(
                [
                    df.iloc[:0],
                    pd.DataFrame([self.columnDiscription], columns=df.columns),
                    df.iloc[0:],
                ],
            )

        if len(self.unwantedRows) != 0:
            headers = df.columns.tolist()
            df.columns = self.header_row

            df = pd.concat(
                [df.iloc[:0], pd.DataFrame([headers], columns=df.columns), df.iloc[0:]],
            )

            for i in range(len(self.unwantedRows)):
                df = pd.concat(
                    [
                        df.iloc[:i],
                        pd.DataFrame([self.unwantedRows[i]], columns=df.columns),
                        df.iloc[i:],
                    ],
                )

            df = pd.concat(
                [
                    df.iloc[: len(self.unwantedRows) + 1],
                    pd.DataFrame([self.columnDiscription], columns=df.columns),
                    df.iloc[len(self.unwantedRows) + 1 :],
                ],
            )

        # folder for all results
        pre_res_folder = os.path.join("results", self.data_name)

        res_folder = os.path.join(pre_res_folder, self.dataProvider)

        # path for anonymized datasets
        self.anon_folder = res_folder
        self.fold = folder_path

        # name for result file
        self.resultFilename = os.path.join(self.anon_folder, self.fileName)

        os.makedirs(self.anon_folder, exist_ok=True)

        df.to_excel(self.resultFilename, sheet_name=sheat, index=False)

    def xlsx_to_excel(self, excel_sheets) -> None:
        output_path = os.path.join(self.fold, self.excelName + "_anonymized" + ".xls")
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
                    # Read the Excel file
                    df = pd.read_excel(os.path.join(self.anon_folder, xlsx_file))
                    df.to_excel(excel_writer, sheet_name=file_name, index=False)

        # Save the Excel file
        excel_writer._save()

    def delete_files(self, names_to_delete) -> None:
        # Get the list of files in the folder
        files = os.listdir(self.anon_folder)

        # Iterate through each file in the folder
        for file_name in files:
            # Check if the file is an xlsx file and contains any of the specified names
            if file_name.endswith(".xlsx") and any(
                name in file_name for name in names_to_delete
            ):
                # Construct the full path to the file
                file_path = os.path.join(self.anon_folder, file_name)

                # Delete the file
                os.remove(file_path)

        shutil.rmtree("results/")


def exec_anonymization(excel_path, folder_path) -> None:
    dataProvider = excel_path.split("/")[-3]

    excel_Sheats = {
        "General info": "General_info",
        "Timepoints": "Timepoints",
        "Baseline": "Baseline",
        "Histology - Mutations": "Histology_Mutations",
        "Treatment": "Treatment",
        "Lab Results": "Lab_Results",
    }
    namesToDelete = {
        "General_info",
        "Timepoints",
        "Baseline",
        "Histology_Mutations",
        "Treatment",
        "Lab_Results",
    }

    sheats_names = pd.ExcelFile(excel_path).sheet_names

    excel_file_name = os.path.splitext(os.path.basename(excel_path))[0]

    for sheat_name in sheats_names:
        sheat = excel_Sheats[sheat_name]
        anonymizer = Anonymizer(
            excel_path=excel_path,
            sheat_Name=sheat,
            data_name=excel_file_name,
            sheat=sheat_name,
            folder_path=folder_path,
            data_provider=dataProvider,
        )

    anonymizer.xlsx_to_excel(excel_sheets=excel_Sheats)
    anonymizer.delete_files(names_to_delete=namesToDelete)


def anonymize_excel(folder_path: str) -> None:
    folder_path = os.path.join(folder_path, "data")
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
            file_path = os.path.join(folder_path, filename)
            exec_anonymization(file_path, folder_path)


def main_cli() -> None:
    import fire

    fire.Fire(anonymize_excel)


if __name__ == "__main__":
    anonymize_excel("prm/samples/valab")
