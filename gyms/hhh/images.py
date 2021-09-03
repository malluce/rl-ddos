import time
import numpy as np

from gyms.hhh.label import Label

ADDRESS_SPACE = 16  # the size of the address space, hierarchy levels are [ADDRESS_SPACE,32]
IMAGE_SIZE = 15  # 8  # the image size (pixels, width=height) in multiples of ADDRESS_SPACE+1

# if the maximum pixel value is at least SQUASH_THRESHOLD times larger than the 2nd smallest value, then
# values will be squashed together with log (smallest value will (likely) be 0, so take 2nd smallest)
SQUASH_THRESHOLD = 1


def generate_hhh_image(hhh_algo, img_size=IMAGE_SIZE, address_space=ADDRESS_SPACE, squash_threshold=SQUASH_THRESHOLD):
    start = time.time()

    img_size_px = img_size * (ADDRESS_SPACE + 1)
    max_addr = 2 ** address_space - 1

    # query HHH for all undiscounted counters, filter hierarchy to be in address space
    res = hhh_algo.query_all()
    res = np.asarray(res)
    res = res[res[:, 0] >= address_space, :]

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
    bounds = np.linspace(0, max_addr, num=img_size_px, dtype=np.int)

    # init output image (x-axis=address bins, y-axis=hierarchy levels)
    image = np.zeros((len(split_by_level), img_size_px), dtype=np.float64)

    # iterate over all hierarchy levels and the corresponding item lists
    for level, l in enumerate(split_by_level):
        # bin the item lists according to the IP address
        l_binned = np.digitize(l[:, 0], bins=bounds)

        shifted_level = level + address_space  # in [ADDRESS_SPACE, 32]
        if shifted_level != 32:  # if not at the bottom, consider possible adjacent bins in subnet
            subnet_size = Label.subnet_size(level + address_space)
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

                if last_bin == len(bounds) - 1:  # also increment last bin if on the right edge
                    image[level, last_bin] += count
            else:  # increment all bins covered by the current item
                # also increment last bin if on the right edge
                # otherwise exclude it (i.e. range(first_bin, last_bin)) to avoid duplicate additions of count
                if last_bin == len(bounds) - 1:
                    image[level, last_bin] += count

                for bin_to_increment in range(first_bin, last_bin):
                    image[level, bin_to_increment] += count

    # if needed squash distant values together to increase visibility of smaller IP sources
    second_smallest_value = np.partition(np.unique(image.flatten()), 1)[1]
    print(f'ratio={np.max(image) / second_smallest_value}')
    if np.max(image) / second_smallest_value >= squash_threshold:
        image = np.log(image, where=np.invert(np.isclose(image, 0.0)))

    # normalize values to be in  [0,1]
    image = image / np.max(image)

    # stretch y-axis to match the x-axis (square image)
    image = np.repeat(image, img_size, axis=0)

    # output (width, height, channels=1)
    image = np.expand_dims(image, 2)
    print(f'[hhh] query_all and CNN image (shape {image.shape}) build time={time.time() - start}')
    return image


def generate_filter_image(hhh_query_result, img_size=IMAGE_SIZE, address_space=ADDRESS_SPACE):
    start = time.time()

    img_size_px = img_size * (ADDRESS_SPACE + 1)
    max_addr = 2 ** address_space - 1

    #
    rules = np.asarray(list(map(lambda x: (x.id, x.len), hhh_query_result)))
    rules = rules[rules[:, 1] >= address_space, :]

    # the bounds of IMAGE_SIZE many bins to separate the address space (x-axis)
    bounds = np.linspace(0, max_addr, num=img_size_px, dtype=np.int)

    # init output image (x-axis=address bins, y-axis=hierarchy levels)
    image = np.zeros((address_space + 1, img_size_px), dtype=np.int)

    # iterate over all rules
    for ip, level in rules:

        if level != 32:  # if not at the bottom, a range of addresses is blocked
            subnet_size = Label.subnet_size(level)
            end_address = ip + subnet_size
        else:  # if at the bottom, only the current IP is blocked
            end_address = ip

        # set image pixels to 1 for blocked ranges of the address space, 0 (default) otherwise
        first_bin = np.digitize(ip, bins=bounds) - 1
        last_bin = np.digitize(end_address, bins=bounds) - 1

        shifted_level = level - address_space
        if first_bin == last_bin:
            image[shifted_level:, first_bin] = 1
        else:
            for blocked_bin in range(first_bin, last_bin + 1):
                image[shifted_level:, blocked_bin] = 1

    # stretch y-axis to match the x-axis (square image)
    image = np.repeat(image, img_size, axis=0)

    # output should be (width, height, channels=1)
    image = np.expand_dims(image, 2)
    print(f'[filter] CNN image (shape {image.shape}) build time={time.time() - start}')
    return image
