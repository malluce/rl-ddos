import time
import numpy as np

from gyms.hhh.label import Label

IMAGE_SIZE = 128  # the image size (width=height) in pixels
ADDRESS_SPACE = 16  # the size of the address space, hierarchy levels are [ADDRESS_SPACE,32]


def generate_hhh_image(hhh_algo, img_size=IMAGE_SIZE, address_space=ADDRESS_SPACE):
    start = time.time()

    max_addr = 2 ** ADDRESS_SPACE - 1

    # query HHH for all undiscounted counters, filter hierarchy to be in address space
    res = hhh_algo.query_all()
    res = np.asarray(res)
    res = res[res[:, 0] >= ADDRESS_SPACE, :]

    if np.max(res[:, 1]) > max_addr:
        raise ValueError(
            f'HHH algo contains addresses larger than currently configured MAX_ADDR ({np.max(res[:, 1])}' > max_addr
        )

    # unique hierarchy levels (should be [ADDRESS_SPACE..32])
    unique_indices = np.unique(res[:, 0], return_index=True)[1]

    # 33-ADDRESS_SPACE lists in ascending order of hierarchy levels (start with ADDRESS_SPACE, end with 32)
    # each list has shape num_items(level) X 2; columns are IP, count
    split_by_level = np.split(res[:, 1:], np.sort(unique_indices)[1:])[::-1]

    # the bounds of IMAGE_SIZE many bins to separate the address space (x-axis)
    bounds = np.linspace(0, max_addr, num=IMAGE_SIZE, dtype=np.int)

    # init output image (x-axis=address bins, y-axis=hierarchy levels)
    image = np.zeros((len(split_by_level), IMAGE_SIZE), dtype=np.float64)

    # iterate over all hierarchy levels and the corresponding item lists
    for level, l in enumerate(split_by_level):
        # bin the item lists according to the IP address
        l_binned = np.digitize(l[:, 0], bins=bounds)

        shifted_level = level + ADDRESS_SPACE  # in [ADDRESS_SPACE, 32]
        if shifted_level != 32:  # if not at the bottom, consider possible adjacent bins in subnet
            subnet_size = Label.subnet_size(level + ADDRESS_SPACE)
            l_end = l[:, 0] + subnet_size
            l_end_binned = np.digitize(l_end, bins=bounds)
        else:  # if at the bottom, only consider the bin which contains the current IP
            l_end_binned = l_binned

        # increment the image pixels for each level and bin according the item list's count
        for bin_index, count in enumerate(l[:, 1]):
            first_bin = l_binned[bin_index] - 1
            last_bin = l_end_binned[bin_index] - 1
            if first_bin == last_bin:  # only increment current bin
                image[level, first_bin] += count
            else:  # increment all bins covered by the current item
                for bin_to_increment in range(first_bin, last_bin):
                    image[level, bin_to_increment] += count

    # normalize values to be in  [0,1]
    image = image / np.max(image)

    # stretch y-axis to match the x-axis (square image)
    image = np.repeat(image, IMAGE_SIZE / ADDRESS_SPACE, axis=0)
    print(f'query_all and CNN image build time={time.time() - start}')
    # TODO img=log(img) to increase clarity for distant values? (image = np.log(image, where=image != 0))

    return image
