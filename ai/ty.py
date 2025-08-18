import random
import uuid

import datetime

START_YEAR = 2010
# START_YEAR = 2025

TODAY = datetime.date.today()

def random_date(start_year=START_YEAR, end_date=TODAY):
    """Generate a random date between a start year and today."""
    start_date = datetime.date(start_year, 1, 1)
    time_between_dates = (end_date - start_date).days
    random_days = random.randint(0, time_between_dates)
    return start_date + datetime.timedelta(days=random_days)

# random_date()
print(random_date())