from datetime import datetime


def get_seconds():
    now = datetime.now()
    return now.strftime("%S")

def get_year_month():
    now = datetime.now()
    return now.strftime("%Y-%m")

def get_month_day():
    now = datetime.now()
    return now.strftime("%m-%d")


def get_year_month_day_hour_minute():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")


def get_year_month_day_hour_minute_second():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def get_day_month_year():
    now = datetime.now()
    return now.strftime("%d-%m-%Y")


def get_day_month_year_hour_minute():
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M")


def get_day_month_year_hour_minute_second():
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")
