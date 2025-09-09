"""
This module loads subspecies names and country mappings from a CSV and provides
functions to query which bumblebee subspecies occur in a given country.
Used at prediction time for geographic context.

Functions
---------
load_country_data(csv_path=SUBSPECIES_COUNTRIES_CSV)
    Load subspecies–country mappings and subspecies names from CSV into dict.
subspecies_seen_in(country_code, by_species, names)
    Return all subspecies observed in a given country, with common and
    scientific names.
"""

import pandas as pd
from bee_tector.config import SUBSPECIES_COUNTRIES_CSV

def load_country_data(csv_path=SUBSPECIES_COUNTRIES_CSV):
    """
    Load subspecies countries and scientific names from CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the CSV file with country and subspecies info

    Returns
    -------
    by_species : dict[str, set[str]]
        Dictionary mapping subspecies class names to the set of country codes
        where they are observed.
    names : dict[str, dict[str, str]]
        Dictionary mapping subspecies class names to their common and
        scientific names, {"common_name", "scientific_name"}.
    """
    df = pd.read_csv(csv_path)

    by_species = (
        df.groupby("class_name")["country_code"]
          .apply(set)
          .to_dict()
    )

    names = (
        df.drop_duplicates("class_name")
          .set_index("class_name")[["common_name", "scientific_name"]]
          .to_dict(orient="index")
    )

    return by_species, names

def subspecies_seen_in(country_code, by_species, names):
    """
    Get all subspecies observed in a given country.

    Parameters
    ----------
    country_code : str or None
        ISO 2-letter country code (if None or empty, returns an empty list).
    by_species : dict[str, set[str]]
        Mapping from subspecies class names to the set of country codes
        where they are observed (as returned by load_country_data).
    names : dict[str, dict[str, str]]
        Metadata for each subspecies, mapping class name to
        {"common_name": str, "scientific_name": str}.

    Returns
    -------
    list of dict
        Sorted list of subspecies dictionaries with:
        - "class_name" : str
        - "common_name" : str
        - "scientific_name" : str
        sorted alphabetically by common name.
    """
    if not country_code:
        return []

    c_code = str(country_code).upper()
    results = []

    for subspecies, countries in by_species.items():
        if c_code in countries:
            info = names.get(subspecies, {})
            results.append({
                "class_name": subspecies,
                "common_name": info.get("common_name", subspecies),
                "scientific_name": info.get("scientific_name", "")
            })

    return sorted(results, key=lambda x: x["common_name"])
