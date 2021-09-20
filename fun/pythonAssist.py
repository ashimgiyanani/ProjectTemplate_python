# -*- coding: utf-8 -*-
import numpy as np
import datetime

class struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


# function to check if a variable exists in local or global workspace
def exists(var):
     var_exists = var in locals() or var in globals()
     return var_exists
# Syntax:
#    exists("variable_name")
# Output:
#    True or False

def myround(x, base=5):
# function to round the x value to the nearest muliple of base
# returns 9 for myround(10.25,3) -> 3*3 = 9
    return base * np.round(x/base)

def now():
# returns the current datetime
#	import datetime
	return datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')

def today():
    from datetime import datetime
    today =  datetime.today().strftime("%d.%m.%Y")
    return today


def FnClosestMember(values, array):
# function to find the closest value for every 'value' in 'array'', size(values) > size(array)
	import numpy as np
	import scipy

	values = -5 + 35*np.random.random((5,1));
	array = np.linspace(0,25,6);
	#make sure that array is a numpy array
	array = np.array(array)
	values = np.array(values)

	# if soreted arrays are necessary
	# idx_sorted = np.argsort(array)
	# sorted_array = np.array(array[idx_sorted])

	# get insert positions
	idxs = np.searchsorted(array, values, side="left")

	# find indexes where previous index is closer
	prev_idx_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
	idxs[prev_idx_less] -= 1
	close_dist = (values - array[idxs])
	close_idx = idxs
	close_values = array[idxs]
	return close_dist, close_idx, close_values

def remove_exponent(x):
# removes the exponent and proves a decimal representation of a number i.e. removes trailing zeros
	from decimal import Decimal
	d = Decimal(x)
	return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

# Example:
# values = -5 + 35*rand(1,100);
# array = 0:5:25;
# close_dist, close_idx, close_value = FnClosestMember(values, array)
      
