from enum import Enum


class MeanType(str, Enum):
    YEAR = "MEAN FOR EVERY YEAR"
    MONTH = "MEAN FOR EVERY MONTH"
    QUATER = "MEAN FOR EVERY 3 MONTHS"
    WEEK = "MEAN OVER A WEEK"
