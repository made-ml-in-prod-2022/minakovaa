import random

import pandas as pd
from faker import Faker
from faker.providers import DynamicProvider

# categorical features providers
sex_provider = DynamicProvider(
    provider_name="sex",
    elements=[0, 1],
)
cp_provider = DynamicProvider(
    provider_name="cp",
    elements=[0, 1, 2, 3],
)
fbs_provider = DynamicProvider(
    provider_name="fbs",
    elements=[0, 1],
)
restecg_provider = DynamicProvider(
    provider_name="restecg",
    elements=[0, 1, 2],
)
exang_provider = DynamicProvider(
    provider_name="exang",
    elements=[0, 1],
)
slope_provider = DynamicProvider(
    provider_name="slope",
    elements=[0, 1, 2],
)
ca_provider = DynamicProvider(
    provider_name="ca",
    elements=[0, 1, 2, 3],
)
thal_provider = DynamicProvider(
    provider_name="thal",
    elements=[0, 1, 2],
)
condition_provider = DynamicProvider(
    provider_name="condition",
    elements=[0, 1],
)

# Numerical features providers


def my_rnd_int_normal_provider(min_el: int, max_el: int, mean: int, std: int):
    rnd_el = int(random.normalvariate(mean, std))

    if rnd_el < min_el:
        return min_el
    elif rnd_el > max_el:
        return max_el

    return rnd_el


def age_gen():
    return my_rnd_int_normal_provider(min_el=29, max_el=90, mean=55, std=9)


def trestbps_gen():
    return my_rnd_int_normal_provider(min_el=94, max_el=200, mean=131, std=17)


def chol_gen():
    return my_rnd_int_normal_provider(min_el=126, max_el=400, mean=247, std=51)


def thalach_gen():
    return my_rnd_int_normal_provider(min_el=71, max_el=202, mean=149, std=22)


def oldpeak_gen():
    return random.expovariate(lambd=1.05)


def create_fake_dataset(n_rows: int, is_include_target=True) -> pd.DataFrame:
    fake = Faker()

    # then add new provider to faker instance
    fake.add_provider(sex_provider)
    fake.add_provider(cp_provider)
    fake.add_provider(fbs_provider)
    fake.add_provider(restecg_provider)
    fake.add_provider(exang_provider)
    fake.add_provider(slope_provider)
    fake.add_provider(ca_provider)
    fake.add_provider(thal_provider)

    fake.add_provider(condition_provider)

    rnd_df = dict()
    rnd_df['sex'] = [fake.sex() for _ in range(n_rows)]
    rnd_df['cp'] = [fake.cp() for _ in range(n_rows)]
    rnd_df['fbs'] = [fake.fbs() for _ in range(n_rows)]
    rnd_df['restecg'] = [fake.restecg() for _ in range(n_rows)]
    rnd_df['exang'] = [fake.exang() for _ in range(n_rows)]
    rnd_df['slope'] = [fake.slope() for _ in range(n_rows)]
    rnd_df['ca'] = [fake.ca() for _ in range(n_rows)]
    rnd_df['thal'] = [fake.thal() for _ in range(n_rows)]

    rnd_df['age'] = [age_gen() for _ in range(n_rows)]
    rnd_df['trestbps'] = [trestbps_gen() for _ in range(n_rows)]
    rnd_df['chol'] = [chol_gen() for _ in range(n_rows)]
    rnd_df['thalach'] = [thalach_gen() for _ in range(n_rows)]
    rnd_df['oldpeak'] = [oldpeak_gen() for _ in range(n_rows)]

    if is_include_target:
        rnd_df['condition'] = [fake.condition() for _ in range(n_rows)]

    return pd.DataFrame(rnd_df)
