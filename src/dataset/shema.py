from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema, Index
from pydantic import BaseModel, Extra

dtypes = {
    "id": Column(int, required=False),
    "site_name": Column(int, required=False),
    "posa_continent": Column(int, required=False),
    "user_location_country": Column(int, required=False),
    "user_location_region": Column(int, required=False),
    "user_location_city": Column(int, required=False),
    "orig_destination_distance": np.float32,
    "user_id": Column(int, required=False),
    "is_mobile": bool,
    "is_package": bool,
    "channel": Column(int, required=False),
    "srch_adults_cnt": Column(int, required=False),
    "srch_children_cnt": Column(int, required=False),
    "srch_rm_cnt": Column(int, required=False),
    "srch_destination_id": Column(int, required=False),
    "srch_destination_type_id": Column(int, required=False),
    "is_booking": bool,
    "cnt": Column(int, required=False),
    "hotel_continent": Column(int, required=False),
    "hotel_country": np.uint16,
    "hotel_market": np.uint16,
    "hotel_cluster": Column(int, required=False),
}

BASE_SCHEMA = DataFrameSchema(
    columns={
        "date_time": Column(datetime),
        "id": Column(int, required=True),
        "site_name": Column(int, required=False),
        "posa_continent": Column(int, required=False),
        "user_location_country": Column(int, required=False),
        "user_location_region": Column(int, required=False),
        "user_location_city": Column(int, required=False),
        "orig_destination_distance": Column(float, required=False),
        "user_id": Column(int, required=False),
        "is_mobile": bool,
        "is_package": bool,
        "channel": Column(int, required=False),
        "srch_ci": Column(datetime),
        "srch_co": Column(datetime),
        "srch_adults_cnt": Column(int, required=False),
        "srch_children_cnt": Column(int, required=False),
        "srch_rm_cnt": Column(int, required=False),
        "srch_destination_id": Column(int, required=False),
        "srch_destination_type_id": Column(int, required=False),
        "is_booking": Column(int, required=False),
        "cnt": Column(int, required=False),
        "hotel_continent": Column(int, required=False),
        "hotel_country": Column(int, required=False),
        "hotel_market": Column(int, required=False),
        "hotel_cluster": Column(int, required=False),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)
