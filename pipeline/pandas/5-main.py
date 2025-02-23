#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
slice = __import__('5-slice').slice

df = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = slice(df)

print(df.tail())
