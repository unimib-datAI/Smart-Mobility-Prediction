import matplotlib.pyplot as plt
import matplotlib.colors
import time
import datetime

def plot_image(image, flow, cmap, vmax, dt):
    # image *= (255.0/image.max())
    im = plt.imshow(image[flow], cmap=cmap, interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(im)
    fl = 'Inflow' if flow==0 else 'Outflow'
    plt.title(dt.strftime(f'%d %B %Y, %H:%M ({fl})'))
    # plt.show()
    # return im


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red","#450903"])


def get_index_of_date(timestamps, T, dt, first01=True):
    '''
    dt: datetime object
    '''

    # check if minute is valid considering T
    valid_minutes = [m for m in range(0, 60, int(60/(T/24)))]
    minute = dt.minute
    assert minute in valid_minutes, f'minute={minute} is not valid with T={T}'

    # build timestamp
    t_per_hour = len(valid_minutes)
    timeslot = dt.hour*t_per_hour + valid_minutes.index(minute)
    if (first01):
        timeslot += 1
    t = dt.strftime('%Y%m%d') + f'{timeslot:02}'
    t = bytes(t, 'utf8')

    # get index
    try:
        return timestamps.index(t)
    except:
        raise Exception(f'{t} not in timestamps')

def timestamp_to_string(timestamp, format):
    t = timestamp.decode('utf8')
    dt = datetime.datetime.strptime(t[:8], '%Y%m%d')
    return dt.strftime(format)
