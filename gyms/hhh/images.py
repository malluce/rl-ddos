import time

import gin
import numpy as np
from gym import spaces

from gyms.hhh.label import Label


@gin.configurable
class ImageGenerator:
    def __init__(self,
                 img_width_px=512,  # the image width (pixels)
                 address_space=16,  # size of the address space=(img height-1), hierarchy levels are [ADDRESS_SPACE,32]
                 # if the
                 # maximum HHH pixel value is at least SQUASH_THRESHOLD times larger than the 2nd smallest value,
                 # then values will be squashed together with log (smallest value will (likely) be 0, so take 2nd
                 # smallest)
                 hhh_squash_threshold=1,
                 max_pixel_value=255  # pixel values are normalized to [0,1] and then multiplied with this value
                 ):
        self.img_width_px = img_width_px
        self.address_space = address_space
        self.hhh_squash_threshold = hhh_squash_threshold
        self.max_pixel_value = max_pixel_value

    def get_hhh_img_spec(self):
        # create dummy input for image gen, feed it through img gen and return img spec
        class DummyHHHAlg:
            def __init__(self, addr_space):
                self.addr_space = addr_space

            def query_all(self):
                return np.array([[x, 0, x] for x in range(self.addr_space, 33)])

        shape = self.generate_hhh_image(DummyHHHAlg(self.address_space)).shape
        return self._shape_to_gym_spec(shape)

    def get_filter_img_spec(self):
        # create dummy filter rule, feed it through img gen and return img spec
        class DummyFilterRule:
            def __init__(self):
                self.id = 0
                self.len = 32

        dummy_hhh_query_result = [DummyFilterRule()]
        shape = self.generate_filter_image(dummy_hhh_query_result).shape
        return self._shape_to_gym_spec(shape)

    def _shape_to_gym_spec(self, shape):
        return spaces.Box(0.0, self.max_pixel_value, shape=shape, dtype=np.float32)

    def generate_hhh_image(self, hhh_algo):
        # start = time.time()

        max_addr = 2 ** self.address_space - 1

        # query HHH for all undiscounted counters, filter hierarchy to be in address space
        res = hhh_algo.query_all()
        res = np.asarray(res)
        res = res[res[:, 0] >= self.address_space, :]

        if np.max(res[:, 1]) > max_addr:
            raise ValueError(
                f'HHH algo contains addresses larger than currently configured MAX_ADDR ({np.max(res[:, 1])}' > max_addr
            )

        # unique hierarchy levels (should be [ADDRESS_SPACE..32])
        unique_indices = np.unique(res[:, 0], return_index=True)[1]

        # 33-ADDRESS_SPACE lists in ascending order of hierarchy levels (start with ADDRESS_SPACE, end with 32)
        # each list has shape num_items(level) X 2; columns are IP, count
        split_by_level = np.split(res[:, 1:], np.sort(unique_indices)[1:])[::-1]

        # the bounds of the bins to separate the address space (x-axis)
        bounds = np.linspace(0, max_addr, num=self.img_width_px, dtype=np.int)

        # init output image (x-axis=address bins, y-axis=hierarchy levels)
        image = np.zeros((len(split_by_level), self.img_width_px), dtype=np.float32)

        # iterate over all hierarchy levels and the corresponding item lists
        for level, l in enumerate(split_by_level):
            # bin the item lists according to the IP address
            l_binned = np.digitize(l[:, 0], bins=bounds)

            shifted_level = level + self.address_space  # in [ADDRESS_SPACE, 32]
            if shifted_level != 32:  # if not at the bottom, consider possible adjacent bins in subnet
                subnet_size = Label.subnet_size(level + self.address_space)
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
        # print(f'ratio={np.max(image) / second_smallest_value}')
        if np.max(image) / second_smallest_value >= self.hhh_squash_threshold:
            image = np.log(image, where=np.invert(np.isclose(image, 0.0)))

        # normalize values to be in  [0,1]
        image = image / np.max(image) * self.max_pixel_value

        # output (height, width, channels=1)
        image = np.expand_dims(image, 2)
        # print(f'[hhh] query_all and CNN image (shape {image.shape}) build time={time.time() - start}')
        return image

    def generate_filter_image(self, hhh_query_result):
        # start = time.time()

        max_addr = 2 ** self.address_space - 1

        #
        rules = np.asarray(list(map(lambda x: (x.id, x.len), hhh_query_result)))

        # the bounds of the bins to separate the address space (x-axis)
        bounds = np.linspace(0, max_addr, num=self.img_width_px, dtype=np.int)

        # init output image (x-axis=address bins, y-axis=hierarchy levels)
        image = np.zeros((self.address_space + 1, self.img_width_px), dtype=np.float32)

        if rules.size == 0:  # if no rules created, return all zero image
            # output should be (height, width, channels=1)
            image = np.expand_dims(image, 2)
            return image

        rules = rules[rules[:, 1] >= self.address_space, :]

        # iterate over all rules
        for ip, level in rules:

            if level != 32:  # if not at the bottom, a range of addresses is blocked
                subnet_size = Label.subnet_size(level)
                end_address = ip + subnet_size
            else:  # if at the bottom, only the current IP is blocked
                end_address = ip

            # set image pixels to max_pixel_value for blocked ranges of the address space, 0 (default) otherwise
            first_bin = np.digitize(ip, bins=bounds) - 1
            last_bin = np.digitize(end_address, bins=bounds) - 1

            shifted_level = level - self.address_space
            if first_bin == last_bin:
                image[shifted_level:, first_bin] = self.max_pixel_value
            else:
                for blocked_bin in range(first_bin, last_bin + 1):
                    image[shifted_level:, blocked_bin] = self.max_pixel_value

        # output should be (height, width, channels=1)
        image = np.expand_dims(image, 2)
        # print(f'[filter] CNN image (shape {image.shape}) build time={time.time() - start}')
        return image
