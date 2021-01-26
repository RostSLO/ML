"""
Created on January 22, 2021

@author rboruk
"""

import pandas as pd


class SearchPeak:

    def __init__(self, f):
        self.res = False
        self.f = f
        self.res = self._getIncreaseDecrease(self.f)

    def __call__(self, *args, **kwargs):
        return self.res

    def _getIncreaseDecrease(self, f):
        diffColumn = []
        # try:
        # read data to the array
        data = pd.read_csv(f + '.csv')
        # get Open and Close price and find the increase\decrease per day
        for index, row in data.iterrows():
            diffColumn.append(100 * ((row['Close'] / row['Open'])-1))
        # add difference as a column to the pandas.DataFrame
        data['Difference'] = diffColumn
        # format Difference column to be a %
        data['Difference'] = pd.Series(["{0:.2f}%".format(val) for val in data['Difference']], index=data.index)
        # write updated data to the new file
        data.to_csv(f + 'Diff.csv')
        return True
        # except:
        #    print("Exception")
        #    return False
