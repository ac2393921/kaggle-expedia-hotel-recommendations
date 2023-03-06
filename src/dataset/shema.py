import numpy as np
from pandera import SchemaModel
from pandera.typing import Bool, DataFrame, DateTime, Object, Series
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


class BaseSchema(SchemaModel):
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
