from typing import Any

import numpy as np
import pandas as pd
from pandera import Column, Field, SchemaModel, SeriesSchema
from pandera.typing import Bool, DataFrame, Object, Series
from pydantic import BaseModel

DTYPES = {
    "site_name": np.uint8,
    "posa_continent": np.uint8,
    "user_location_country": np.uint16,
    "user_location_region": np.uint16,
    "user_location_city": np.uint16,
    "orig_destination_distance": np.float32,
    "user_id": np.uint32,
    "is_mobile": bool,
    "is_package": bool,
    "channel": np.uint8,
    "srch_adults_cnt": np.uint8,
    "srch_children_cnt": np.uint8,
    "srch_rm_cnt": np.uint8,
    "srch_destination_id": np.uint32,
    "srch_destination_type_id": np.uint8,
    "is_booking": bool,
    "cnt": np.uint64,
    "hotel_continent": np.uint8,
    "hotel_country": np.uint16,
    "hotel_market": np.uint16,
    "hotel_cluster": np.uint8,
}


class HotelTrainBaseSchema(SchemaModel):
    date_time: Series[Object]
    site_name: Series[np.uint8]
    posa_continent: Series[np.uint8]
    user_location_country: Series[np.uint16]
    user_location_region: Series[np.uint16]
    user_location_city: Series[np.uint16]
    orig_destination_distance: Series[np.float32]
    user_id: Series[np.uint32]
    is_mobile: Series[Bool]
    is_package: Series[Bool]
    channel: Series[np.uint8]
    srch_ci: Series[Object]
    srch_co: Series[Object]
    srch_adults_cnt: Series[np.uint8]
    srch_children_cnt: Series[np.uint8]
    srch_rm_cnt: Series[np.uint8]
    srch_destination_id: Series[np.uint32]
    srch_destination_type_id: Series[np.uint8]
    is_booking: Series[Bool]
    cnt: Series[np.uint64]
    hotel_continent: Series[np.uint8]
    hotel_country: Series[np.uint16]
    hotel_market: Series[np.uint16]
    hotel_cluster: Series[np.uint8]


class HotelTrainSchema(SchemaModel):
    date_time: Series[Object]
    site_name: Series[np.uint8]
    posa_continent: Series[np.uint8]
    user_location_country: Series[np.uint16]
    user_location_region: Series[np.uint16]
    user_location_city: Series[np.uint16]
    orig_destination_distance: Series[np.float32]
    user_id: Series[np.uint32]
    is_mobile: Series[Bool]
    is_package: Series[Bool]
    channel: Series[np.uint8]
    srch_ci: Series[Object]
    srch_co: Series[Object]
    srch_adults_cnt: Series[np.uint8]
    srch_children_cnt: Series[np.uint8]
    srch_rm_cnt: Series[np.uint8]
    srch_destination_id: Series[np.uint32]
    srch_destination_type_id: Series[np.uint8]
    is_booking: Series[Bool]
    cnt: Series[np.uint64]
    hotel_continent: Series[np.uint8]
    hotel_country: Series[np.uint16]
    hotel_market: Series[np.uint16]


class HotelTestBaseSchema(SchemaModel):
    date_time: Series[Object]
    site_name: Series[np.uint8]
    posa_continent: Series[np.uint8]
    user_location_country: Series[np.uint16]
    user_location_region: Series[np.uint16]
    user_location_city: Series[np.uint16]
    orig_destination_distance: Series[np.float32]
    user_id: Series[np.uint32]
    is_mobile: Series[Bool]
    is_package: Series[Bool]
    channel: Series[np.uint8]
    srch_ci: Series[Object]
    srch_co: Series[Object]
    srch_adults_cnt: Series[np.uint8]
    srch_children_cnt: Series[np.uint8]
    srch_rm_cnt: Series[np.uint8]
    srch_destination_id: Series[np.uint32]
    srch_destination_type_id: Series[np.uint8]
    is_booking: Series[Bool]
    cnt: Series[np.uint64]
    hotel_continent: Series[np.uint8]
    hotel_country: Series[np.uint16]
    hotel_market: Series[np.uint16]
    hotel_cluster: Series[np.uint8]


class HotelTestSchema(SchemaModel):
    date_time: Series[Object]
    site_name: Series[np.uint8]
    posa_continent: Series[np.uint8]
    user_location_country: Series[np.uint16]
    user_location_region: Series[np.uint16]
    user_location_city: Series[np.uint16]
    orig_destination_distance: Series[np.float32] = Field(nullable=True, coerce=True)
    user_id: Series[np.uint32]
    is_mobile: Series[Bool]
    is_package: Series[Bool]
    channel: Series[np.uint8]
    srch_ci: Series[Object] = Field(nullable=True, coerce=True)
    srch_co: Series[Object] = Field(nullable=True, coerce=True)
    srch_adults_cnt: Series[np.uint8]
    srch_children_cnt: Series[np.uint8]
    srch_rm_cnt: Series[np.uint8]
    srch_destination_id: Series[np.uint32]
    srch_destination_type_id: Series[np.uint8]
    hotel_continent: Series[np.uint8]
    hotel_country: Series[np.uint16]
    hotel_market: Series[np.uint16]


class TrainSchema(SchemaModel):
    date_time: Series[Object]
    site_name: Series[np.uint8]
    posa_continent: Series[np.uint8]
    user_location_country: Series[np.uint16]
    user_location_region: Series[np.uint16]
    user_location_city: Series[np.uint16]
    orig_destination_distance: Series[np.float32]
    user_id: Series[np.uint32]
    is_mobile: Series[Bool]
    is_package: Series[Bool]
    channel: Series[np.uint8]
    srch_ci: Series[Object]
    srch_co: Series[Object]
    srch_adults_cnt: Series[np.uint8]
    srch_children_cnt: Series[np.uint8]
    srch_rm_cnt: Series[np.uint8]
    srch_destination_id: Series[np.uint32]
    srch_destination_type_id: Series[np.uint8]
    is_booking: Series[Bool]
    cnt: Series[np.uint64]
    hotel_continent: Series[np.uint8]
    hotel_country: Series[np.uint16]
    hotel_market: Series[np.uint16]


# class TestSchema(SchemaModel):
#     hotel_cluster: Series[np.uint8]


TARGET_SCHEMA = SeriesSchema(np.uint8)


class RawData(BaseModel):
    train_data: DataFrame[TrainSchema]
    target: Any


class TestData(BaseModel):
    id: Any
    x: Any
    # target: Series
    # target: np.uint8
    # target: Any


class SplitData(BaseModel):
    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any


# class SplitData(BaseModel):
#     x_train: DataFrame[TrainSchema]
#     x_test: DataFrame[TrainSchema]
#     y_train: Any
#     y_test: Any
