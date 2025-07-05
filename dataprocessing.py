import pandas as pd
from pathlib import Path
from typing import Union

def load_and_process_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a dataset from a file and process it by combining 'Patient' and 'Description'
    columns into a new 'text' column.

    Supported formats: .csv, .xlsx, .xls, .tsv, .txt (tab/comma delimited)

    Parameters:
    ----------
    file_path : str or Path
        The path to the file to load.

    Returns:
    -------
    pd.DataFrame
        The processed DataFrame with an additional 'text' column.
    
    Raises:
    ------
    ValueError
        If the file extension is unsupported or required columns are missing.
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()

    # Load data based on file extension
    if file_ext in ['.csv']:
        df = pd.read_csv(file_path)
    elif file_ext in ['.tsv', '.txt']:
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception:
            df = pd.read_csv(file_path, sep=',')  # fallback to comma-separated
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Validate required columns
    required_columns = {'Patient', 'Description'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Perform the task: create 'text' column
    df['text'] = df['Patient'].astype(str) + " " + df['Description'].astype(str)

    return df

